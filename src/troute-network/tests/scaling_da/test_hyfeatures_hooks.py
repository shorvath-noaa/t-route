"""HYFeaturesNetwork side tables for area-scaled streamflow DA (network hooks)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("geopandas")  # HYFeaturesNetwork needs the full troute-network stack
import troute.HYFeaturesNetwork as HYF  # noqa: E402


def test_segment_attributes_flowpath_keys():
    fp = pd.DataFrame(
        {"flowpath_id": ["fp-12", "fp-3"], "total_dasqkm": [5.5, 1.0], "vpuid": ["01", "01"], "other": [1, 2]}
    )
    out = HYF.segment_attributes(fp, "flowpath_id")
    assert out.index.tolist() == [3, 12]
    assert out.index.name == "key" and out.index.dtype == np.int64
    assert list(out.columns) == ["total_dasqkm", "vpuid"]
    assert out["total_dasqkm"].dtype == np.float64
    assert isinstance(out["vpuid"].dtype, pd.CategoricalDtype)
    assert out.loc[12, "total_dasqkm"] == 5.5


def test_segment_attributes_flowline_keys():
    fl = pd.DataFrame({"flowline_id": [9, 7, 7], "total_dasqkm": [2.0, np.nan, 3.0], "vpuid": ["03N"] * 3})
    out = HYF.segment_attributes(fl, "flowline_id")
    assert out.index.tolist() == [7, 9]  # duplicates keep the first row
    assert np.isnan(out.loc[7, "total_dasqkm"])


def test_segment_attributes_missing_columns():
    out = HYF.segment_attributes(pd.DataFrame({"flowpath_id": ["fp-1"]}), "flowpath_id")
    assert out.empty


def test_gage_hydrolocations():
    hl = pd.DataFrame(
        {
            "flowpath_id": ["fp-1", "fp-2", "fp-3"],
            "hl_class": ["gage", "lake", "gage"],
            "hl_reference": ["nwis-01010000| rfc-ABCD1", "wbout-9", "nwis-0102"],
            "vpuid": ["01"] * 3,
            "poi_da_ratio": [1.0, np.nan, 2.0],
            "poi_da_flag": [0, np.nan, 1],
            "geometry": [None] * 3,
        }
    )
    out = HYF.gage_hydrolocations(hl)
    assert out[["flowpath_id", "type", "site"]].values.tolist() == [
        ["fp-1", "nwis", "01010000"],
        ["fp-1", "rfc", "ABCD1"],
        ["fp-3", "nwis", "0102"],
    ]
    assert "geometry" not in out.columns
    assert out["poi_da_flag"].tolist()[2] == 1


def test_gage_hydrolocations_empty():
    assert HYF.gage_hydrolocations(pd.DataFrame()).empty
    assert HYF.gage_hydrolocations(pd.DataFrame({"flowpath_id": ["fp-1"]})).empty
