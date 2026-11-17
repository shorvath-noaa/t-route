"""Configuration of area-scaled streamflow DA (`streamflow_da.streamflow_scaling`)."""

import math

import pytest
from pydantic import ValidationError

from troute.config.compute_parameters import (
    DataAssimilationParameters,
    StreamflowDA,
    StreamflowScalingParams,
    ThetaRegionalization,
)


def da(tmp_path, **streamflow_da):
    return DataAssimilationParameters(usgs_timeslices_folder=str(tmp_path), streamflow_da=streamflow_da)


def test_defaults_are_off_and_complete():
    s = StreamflowDA(streamflow_nudging=True)
    assert s.streamflow_scaling is False
    p = s.streamflow_scaling_parameters
    assert p.theta.default == 0.77
    assert p.theta.by_vpu == {} and p.theta.per_tree_csv is None
    assert p.max_reach_km == 200.0 and p.min_flow_cms == 1e-6
    assert p.holdout_sites_file is None
    assert p.site_types == ["nwis"] and p.extra_stop_segs == []
    assert p.spread_chunk_timesteps is None


def test_dict_round_trip_keeps_existing_keys():
    d = StreamflowDA(streamflow_nudging=True).dict()
    for key in ("streamflow_nudging", "gage_segID_crosswalk_file", "lastobs_file", "diffusive_streamflow_nudging"):
        assert key in d
    assert d["streamflow_scaling"] is False
    assert d["streamflow_scaling_parameters"]["theta"]["default"] == 0.77


def test_scaling_requires_nudging():
    with pytest.raises(ValidationError, match="streamflow_nudging: True"):
        StreamflowDA(streamflow_scaling=True)
    with pytest.raises(ValidationError, match="streamflow_nudging: True"):
        StreamflowDA(streamflow_nudging=False, streamflow_scaling=True)
    assert StreamflowDA(streamflow_nudging=True, streamflow_scaling=True).streamflow_scaling


def test_scaling_needs_timeslice_folder(tmp_path):
    with pytest.raises(ValidationError, match="usgs_timeslices_folder"):
        DataAssimilationParameters(streamflow_da={"streamflow_nudging": True, "streamflow_scaling": True})
    assert da(tmp_path, streamflow_nudging=True, streamflow_scaling=True).streamflow_da.streamflow_scaling
    # nudging alone keeps its existing behavior (no new requirement)
    DataAssimilationParameters(streamflow_da={"streamflow_nudging": True})


@pytest.mark.parametrize("bad", [0.0, -0.5, math.inf, math.nan])
def test_theta_default_must_be_finite_positive(bad):
    with pytest.raises(ValidationError, match="theta.default"):
        ThetaRegionalization(default=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_theta_by_vpu_values_must_be_finite_positive(bad):
    with pytest.raises(ValidationError, match="by_vpu"):
        ThetaRegionalization(by_vpu={"01": bad})


def test_theta_by_vpu_keys_stay_strings():
    t = ThetaRegionalization(by_vpu={"01": 0.6, "03N": 0.5})
    assert set(t.by_vpu) == {"01", "03N"}
    with pytest.raises(ValidationError, match="empty key"):
        ThetaRegionalization(by_vpu={" ": 0.6})


@pytest.mark.parametrize("field", ["max_reach_km", "min_flow_cms"])
@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_positive_parameters(field, bad):
    with pytest.raises(ValidationError, match=field):
        StreamflowScalingParams(**{field: bad})


def test_site_types_non_empty():
    with pytest.raises(ValidationError, match="site_types"):
        StreamflowScalingParams(site_types=[])
    assert StreamflowScalingParams(site_types=["nwis", "rfc"]).site_types == ["nwis", "rfc"]


def test_chunk_timesteps_non_negative():
    pytest.raises(ValidationError, StreamflowScalingParams, spread_chunk_timesteps=-1)
    assert StreamflowScalingParams(spread_chunk_timesteps=0).spread_chunk_timesteps == 0


def test_misspelled_keys_are_errors():
    for model, typo in ((StreamflowScalingParams, {"max_reach": 100}),
                        (StreamflowScalingParams, {"holdout_file": "x.txt"}),
                        (ThetaRegionalization, {"per_tree": "x.csv"})):
        pytest.raises(ValidationError, model, **typo)


# NGWPC options not ported: (key, NGWPC default, a non-default value)
DEFERRED = [
    ("innovation_spread_h", 0.0, 6.0),
    ("travel_time_lag", False, True),
    ("lag_window_h", 48.0, 24.0),
    ("synthetic_obs_factor", None, 1.2),
    ("synthetic_obs_baseline", None, "/tmp/baseline"),
]


@pytest.mark.parametrize("key,default,other", DEFERRED)
def test_deferred_ngwpc_options(key, default, other):
    # NGWPC configs that leave these at their defaults translate unchanged
    p = StreamflowScalingParams(**{key: default})
    assert key not in p.dict()
    with pytest.raises(ValidationError, match="not supported"):
        StreamflowScalingParams(**{key: other})


def test_ngwpc_style_block_translates(tmp_path):
    csv = tmp_path / "theta.csv"
    csv.write_text("gage_id,theta\n01104000,0.6\n")
    holdout = tmp_path / "holdout.txt"
    holdout.write_text("01105000\n")
    block = dict(
        theta={"default": 0.7, "by_vpu": {"01": 0.65}, "per_tree_csv": str(csv)},
        holdout_sites_file=str(holdout),
        min_flow_cms=1e-5,
        max_reach_km=150.0,
        spread_chunk_timesteps=12,
    )
    block.update((key, default) for key, default, _ in DEFERRED[:2])  # explicit NGWPC defaults
    # NGWPC required streamflow_nudging False here; this version requires True
    d = da(tmp_path, streamflow_nudging=True, streamflow_scaling=True, streamflow_scaling_parameters=block)
    p = getattr(d.streamflow_da, "streamflow_scaling_parameters")
    assert p.theta.default == 0.7 and p.theta.by_vpu == {"01": 0.65}
    assert str(p.theta.per_tree_csv) == str(csv) and str(p.holdout_sites_file) == str(holdout)
    assert (p.min_flow_cms, p.max_reach_km, p.spread_chunk_timesteps) == (1e-5, 150.0, 12)
