"""Scaling-DA network adapter on a real HFv4.0 subset (skipped unless the fixture exists).

The fixture is generated, not committed. Generate it with
`test/tools/hfv4_subset.py upstream` (see `test/tools/README.md`) and point
`TROUTE_HFV4_FIXTURE` at the `.gpkg`:

    TROUTE_HFV4_FIXTURE=/path/hfv4_fixture.gpkg pytest src/troute-network/tests/scaling_da -m hfv4_fixture
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from troute.scaling_da import pack_trees
from troute.scaling_da.forest import DROP_REASONS
from troute.scaling_da.hyfeatures import (
    ADAPTER_DROP_REASONS,
    build_hyfeatures_trees,
    results_seg_to_row,
    routed_segment_ids,
)

FIXTURE = os.environ.get("TROUTE_HFV4_FIXTURE", "")

pytestmark = [
    pytest.mark.hfv4_fixture,
    pytest.mark.skipif(
        not (FIXTURE and Path(FIXTURE).is_file()),
        reason="set TROUTE_HFV4_FIXTURE to an HFv4.0 subset gpkg (test/tools/hfv4_subset.py upstream)",
    ),
]

COLUMNS = {
    "flowpath": {"key": "flowpath_id", "downstream": "flowpath_toid"},
    "flowline": {"key": "flowline_id", "downstream": "flowline_toid"},
}
CHANNEL = {
    "dx": "lengthm", "n": "n", "ncc": "nCC", "s0": "So", "bw": "BtmWdth", "waterbody": "lake_id",
    "tw": "TopWdth", "twcc": "TopWdthCC", "musk": "MusK", "musx": "MusX", "cs": "ChSlp", "alt": "alt",
}


def build_network(gpkg: str, mode: str, qlat_dir: Path):
    """Build a HYFeaturesNetwork the way `main_v04` does (config -> dicts -> network)."""
    pytest.importorskip("geopandas")
    from troute.config import Config
    from troute.HYFeaturesNetwork import HYFeaturesNetwork

    topology = dict(
        supernetwork_parameters={"geo_file_path": gpkg, "columns": {**COLUMNS[mode], **CHANNEL}},
        waterbody_parameters={"break_network_at_waterbodies": True},
    )
    da = dict(
        streamflow_da={"streamflow_nudging": True, "diffusive_streamflow_nudging": False},
        reservoir_da={
            "reservoir_persistence_da": {"reservoir_persistence_usgs": False,
                                         "reservoir_persistence_usace": False},
            "reservoir_rfc_da": {"reservoir_rfc_forecasts": False},
        },
    )
    data = {
        "log_parameters": {"showtiming": False, "log_level": "WARNING"},
        "network_topology_parameters": topology,
        "compute_parameters": {
            "parallel_compute_method": "serial",
            "compute_kernel": "V02-structured",
            "assume_short_ts": True,
            "cpu_pool": 1,
            "restart_parameters": {"start_datetime": "2020-03-01_00:00:00"},
            "hybrid_parameters": {"run_hybrid_routing": False},
            "forcing_parameters": {"qts_subdivisions": 12, "dt": 300, "nts": 288, "max_loop_size": 4,
                                   "qlat_input_folder": str(qlat_dir)},
        },
    }
    data["compute_parameters"].update(data_assimilation_parameters=da)
    cfg = Config.with_strict_mode(**data).dict()
    cp, nt = cfg["compute_parameters"], cfg["network_topology_parameters"]
    return HYFeaturesNetwork(
        nt["supernetwork_parameters"], nt["waterbody_parameters"], cp["data_assimilation_parameters"],
        cp["restart_parameters"], cp, cp["forcing_parameters"], cp["hybrid_parameters"],
        nt["preprocessing_parameters"], cfg["output_parameters"], verbose=False, showtiming=False,
    )


@pytest.fixture(scope="module", params=["flowpath", "flowline"])
def built(request, tmp_path_factory):
    net = build_network(FIXTURE, request.param, tmp_path_factory.mktemp("qlat"))
    return request.param, net, build_hyfeatures_trees(net)


def test_side_tables_do_not_touch_routing_dataframe(built):
    _, net, _ = built
    assert not {"total_dasqkm", "vpuid"} & set(net.dataframe.columns)
    assert {"total_dasqkm", "vpuid"} <= set(net.segment_attributes.columns)
    assert not net.gage_hydrolocations.empty


def test_routed_segments_have_area(built):
    _, net, b = built
    routed = set(routed_segment_ids(net).tolist())
    segs = routed - b.mappings.lake_segments
    missing = segs - set(net.segment_attributes.index.tolist())
    assert not missing, f"{len(missing)} routed segments lack total_dasqkm, e.g. {sorted(missing)[:5]}"


def test_trees_build(built):
    mode, _, b = built
    assert b.result.trees, f"no gage trees in {mode} mode; drops: {b.dropped}"
    print(f"\n{mode}: {len(b.result.trees)} trees, "
          f"{sum(t.n_nodes for t in b.result.trees.values())} nodes, drops {b.dropped}, "
          f"excluded types {b.mappings.excluded_types}, "
          f"network area inversions {b.mappings.n_area_inversions_network}")


def test_every_gage_mapped_or_dropped_with_reason(built):
    _, net, b = built
    nwis = set(net.gage_hydrolocations.loc[net.gage_hydrolocations["type"] == "nwis", "site"].astype(str))
    reasons = set(ADAPTER_DROP_REASONS) | set(DROP_REASONS)
    assert set(b.gage_report["site"]) == nwis
    for site in nwis:
        assert (site in b.result.trees) != (site in b.dropped), site
        if site in b.dropped:
            assert b.dropped[site] in reasons


def test_lake_stops(built):
    _, net, b = built
    wb = b.mappings.lake_segments
    assert wb, "fixture should contain at least one lake"
    rconn = net.reverse_network
    for site, t in b.result.trees.items():
        members = t.seg_ids.tolist()
        assert not set(members) & wb, f"tree {site} enters a waterbody"
        for i, s in enumerate(members):
            lake_up = {int(u) for u in rconn.get(s, ()) if int(u) in wb}
            assert lake_up <= set(t.pruned_of(i).tolist()), f"tree {site}: lake above {s} not pruned"


def test_trees_pack_against_routed_ids(built):
    _, net, b = built
    packed, dropped = pack_trees(b.result.trees, results_seg_to_row(routed_segment_ids(net)))
    assert dropped == {}
    assert packed.rows.size == sum(t.n_nodes for t in b.result.trees.values())
    assert np.unique(packed.rows).size == packed.rows.size
