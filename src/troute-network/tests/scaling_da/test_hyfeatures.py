"""HYFeatures network adapter (troute.scaling_da.hyfeatures) on in-memory networks."""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from troute.scaling_da import pack_trees
from troute.scaling_da.forest import DROP_REASONS as DROP_REASONS_CORE
from troute.scaling_da.hyfeatures import (
    ADAPTER_DROP_REASONS,
    ArrayMapping,
    build_hyfeatures_trees,
    hyfeatures_mappings,
    network_mode,
    results_seg_to_row,
    routed_segment_ids,
)

# Routed graph after lake collapse (upstream -> downstream). Lake 9001 has
# member segments 161 and 162 (not routed); its node replaces them.
#
#   15 -> 14[B] -> 12[rfc R] -\
#                              11 -> 10[A]
#   17 -> (lake 9001) -> 13 --/
#                 18 ---/
#   31[D: not routed]   30 (diffusive mainstem, tailwater 30) [E]
#
EDGES = {11: 10, 12: 11, 13: 11, 14: 12, 15: 14, 9001: 13, 17: 9001, 18: 13}
INCR = {10: 5.0, 11: 4.0, 12: 3.0, 13: 2.0, 14: 6.0, 15: 7.0, 17: 9.0, 18: 1.0, 30: 50.0}
LAKE_MEMBERS = {161: 9001, 162: 9001}


def _total(edges, incr):
    up = {}
    for u, d in edges.items():
        up.setdefault(d, []).append(u)
    tot = {}

    def acc(s):
        if s not in tot:
            tot[s] = incr.get(s, 0.0) + sum(acc(u) for u in up.get(s, ()))
        return tot[s]

    for s in set(edges) | set(edges.values()) | set(incr):
        acc(s)
    return tot


def make_network(mode="flowpath", *, hl_extra=(), gages_extra=None, compute_nodes=False,
                 total_override=None, drop_total=False, hydrolocations=True):
    total = _total(EDGES, INCR)
    total.pop(9001)  # lake node: no hydrofabric segment
    for m in LAKE_MEMBERS:
        total[m] = 20.0
    total.update(total_override or {})
    rconn = {}
    for u, d in sorted(EDGES.items()):
        rconn.setdefault(d, []).append(u)
        rconn.setdefault(u, [])
    rconn[30] = []  # diffusive mainstem segment still present in the graph
    segs = sorted(total)
    seg_attrs = pd.DataFrame(
        {"total_dasqkm": [total[s] for s in segs], "vpuid": pd.Categorical(["01"] * len(segs))},
        index=pd.Index(segs, name="key"),
    )
    if drop_total:
        seg_attrs = seg_attrs.drop(columns="total_dasqkm")
    dataframe = pd.DataFrame({"dx": [1000.0 * (1 + (s % 3)) for s in segs]}, index=pd.Index(segs, name="key"))
    rows = [
        ("fp-10", "nwis-A", 0.0, 0, 37.0),
        ("fp-14", "nwis-B", 3.0, 1, 13.0),
        ("fp-16", "nwis-L", 0.0, 0, 20.0),
        ("fp-99", "nwis-X", 0.0, 0, 5.0),
        ("fp-12", "rfc-R", None, None, None),
        ("fp-31", "nwis-D", None, None, None),
        ("fp-30", "nwis-E", None, None, None),
    ] + list(hl_extra)
    hl = pd.DataFrame(
        {
            "flowpath_id": [r[0] for r in rows],
            "hl_reference": [r[1] for r in rows],
            "vpuid": ["01"] * len(rows),
            "poi_da_ratio": [r[2] for r in rows],
            "poi_da_flag": [r[3] for r in rows],
            "poi_da_ref_sqkm": [r[4] for r in rows],
            "poi_da_ngen_sqkm": [r[4] for r in rows],
        }
    )
    parts = hl["hl_reference"].str.partition("-")
    hl["type"], hl["site"] = parts[0], parts[2]
    gages = {10: "A", 14: "B", 161: "L", 12: "R", 31: "D", 30: "E"}
    gages.update(gages_extra or {})
    key = "flowline_id" if mode == "flowline" else "flowpath_id"
    return SimpleNamespace(
        reverse_network=rconn,
        dataframe=dataframe,
        segment_attributes=seg_attrs,
        gage_hydrolocations=hl if hydrolocations else pd.DataFrame(),
        gages={"gages": gages},
        waterbody_connections=dict(LAKE_MEMBERS),
        compute_parameters={"compute_nodes": compute_nodes},
        supernetwork_parameters={"columns": {"key": key}},
        diffusive_network_data={30: {"mainstem_segs": [30]}},
    )


# ------------------------------------------------------------------ mappings
@pytest.mark.parametrize("mode", ["flowpath", "flowline"])
def test_mappings_both_modes(mode):
    m = hyfeatures_mappings(make_network(mode))
    assert m.mode == mode
    assert m.gage_segments == {"A": 10, "B": 14, "L": 161}  # ordered by segment
    assert list(m.gage_segments) == ["A", "B", "L"]
    assert m.dropped == {"X": "not_in_crosswalk", "D": "not_routed", "E": "not_routed"}
    assert m.excluded_types == {"rfc": 1}
    assert m.lake_segments == {161, 162, 9001}
    assert 30 in m.extra_stop_segs
    assert m.site_vpu == {"A": "01", "B": "01", "L": "01"}
    assert m.drainage_km2[10] == pytest.approx(sum(INCR[s] for s in INCR if s != 30))
    assert m.length_km[11] == pytest.approx(3.0)
    assert set(m.dropped.values()) <= set(ADAPTER_DROP_REASONS)


def test_gage_report_flags_and_status():
    m = hyfeatures_mappings(make_network())
    r = m.gage_report.set_index("site")
    assert sorted(r.index) == ["A", "B", "D", "E", "L", "X"]  # rfc not reported
    assert r.loc["X", "status"] == "not_in_crosswalk"
    assert pd.isna(r.loc["X", "segment"])
    assert r.loc["A", "status"] == "mapped"
    assert r.loc["A", "model_to_ref"] == pytest.approx(r.loc["A", "area_model_sqkm"] / 37.0)
    assert not r.loc["A", "flagged"]
    assert r.loc["B", "flagged"]  # poi_da_flag and relative difference 3.0
    assert not r.loc["D", "flagged"]  # no poi data -> not flagged


def test_poi_da_ratio_is_a_relative_difference():
    # HFv4.0 poi_da_ratio: 0 = agreement. Each rule flags on its own.
    net = make_network()
    hl = net.gage_hydrolocations
    a = hl["site"] == "A"
    hl.loc[a, "poi_da_ratio"] = 0.2  # within 0.25
    assert not hyfeatures_mappings(net).gage_report.set_index("site").loc["A", "flagged"]
    hl.loc[a, "poi_da_ratio"] = 0.3  # relative difference too large
    assert hyfeatures_mappings(net).gage_report.set_index("site").loc["A", "flagged"]
    hl.loc[a, "poi_da_ratio"] = 0.0
    hl.loc[a, "poi_da_ref_sqkm"] = 37.0 / 1.3  # model/reference = 1.3
    assert hyfeatures_mappings(net).gage_report.set_index("site").loc["A", "flagged"]


def test_flagged_gages_are_kept(caplog):
    with caplog.at_level(logging.WARNING, logger="troute.scaling_da.hyfeatures"):
        b = build_hyfeatures_trees(make_network())
    assert "B" in b.result.trees
    assert any("disagree with the hydrofabric drainage area" in rec.message for rec in caplog.records)


def test_drops_are_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="troute.scaling_da.hyfeatures"):
        hyfeatures_mappings(make_network())
    msgs = " ".join(rec.message for rec in caplog.records)
    assert "not_in_crosswalk" not in msgs  # logs the description, not the key
    assert ADAPTER_DROP_REASONS["not_in_crosswalk"] in msgs
    assert ADAPTER_DROP_REASONS["not_routed"] in msgs


def test_site_types_option():
    m = hyfeatures_mappings(make_network(), site_types=("nwis", "rfc"))
    assert m.gage_segments["R"] == 12
    assert m.excluded_types == {}


def test_site_vpu_falls_back_to_hydrolocation():
    net = make_network()
    net.segment_attributes = net.segment_attributes.drop(columns="vpuid")
    net.gage_hydrolocations.loc[net.gage_hydrolocations["site"] == "A", "vpuid"] = "02"
    m = hyfeatures_mappings(net)
    assert m.site_vpu["A"] == "02"


# ------------------------------------------------------------------ trees
def test_trees_stop_at_lake_gages_and_diffusive():
    b = build_hyfeatures_trees(make_network(), max_reach_km=None)
    trees = b.result.trees
    assert sorted(trees) == ["A", "B"]
    a = trees["A"]
    assert set(a.seg_ids.tolist()) == {10, 11, 12, 13, 18}  # rfc segment 12 is not a stop
    i12 = a.seg_ids.tolist().index(12)
    i13 = a.seg_ids.tolist().index(13)
    assert a.pruned_of(i12).tolist() == [14]
    assert a.pruned_of(i13).tolist() == [9001]
    assert set(trees["B"].seg_ids.tolist()) == {14, 15}
    assert b.dropped["L"] == "gage_on_waterbody"
    assert set(b.dropped.values()) <= set(ADAPTER_DROP_REASONS) | set(DROP_REASONS_CORE)
    r = b.gage_report.set_index("site")
    assert r.loc["A", "status"] == "tree" and r.loc["A", "n_tree_segments"] == 5
    assert r.loc["L", "status"] == "gage_on_waterbody" and r.loc["L", "n_tree_segments"] == 0


def test_members_and_pruned_are_routed_nodes():
    net = make_network()
    b = build_hyfeatures_trees(net, max_reach_km=None)
    routed = set(routed_segment_ids(net).tolist())
    for t in b.result.trees.values():
        assert set(t.seg_ids.tolist()) <= routed
        assert set(t.pruned_ids.tolist()) <= routed
    packed, dropped = pack_trees(b.result.trees, results_seg_to_row(routed_segment_ids(net)))
    assert dropped == {}


def test_distance_cap():
    # dx(m) = 1000*(1 + s%3): 10->2 km, 11->3 km, 13->2 km. Node 18 sits 2+3+2 = 7 km up.
    b = build_hyfeatures_trees(make_network(), max_reach_km=6.0)
    a = b.result.trees["A"]
    assert 18 not in a.seg_ids.tolist()
    assert b.result.n_capped["A"] == 1


def test_extra_stop_segs():
    b = build_hyfeatures_trees(make_network(), extra_stop_segs=[13], max_reach_km=None)
    assert set(b.result.trees["A"].seg_ids.tolist()) == {10, 11, 12}


def test_area_inversion_count():
    net = make_network(total_override={15: 1000.0})  # 15 drains to 14 but is larger
    m = hyfeatures_mappings(net)
    assert m.n_area_inversions_network == 1


def test_missing_area_drops_tree():
    net = make_network(total_override={18: float("nan")})
    b = build_hyfeatures_trees(net, max_reach_km=None)
    assert b.dropped["A"] == "nonfinite_area"


# ------------------------------------------------------------------ refusals
def test_compute_nodes_refused():
    with pytest.raises(NotImplementedError, match="compute_nodes"):
        hyfeatures_mappings(make_network(compute_nodes=True))


def test_missing_total_dasqkm():
    with pytest.raises(ValueError, match="total_dasqkm"):
        hyfeatures_mappings(make_network(drop_total=True))


def test_missing_hydrolocations():
    with pytest.raises(ValueError, match="hydrolocations"):
        hyfeatures_mappings(make_network(hydrolocations=False))


# ------------------------------------------------------------------ helpers
def test_array_mapping():
    m = ArrayMapping([5, 1, 3, 5], [50.0, 10.0, 30.0, 99.0])
    assert len(m) == 3 and list(m) == [1, 3, 5]
    assert m[5] == 50.0  # first duplicate wins
    assert m.get(np.int64(3)) == 30.0
    assert m.get(2) is None and m.get(2, -1.0) == -1.0
    assert m.get("x") is None and m.get(2**70) is None
    assert 1 in m and 4 not in m
    with pytest.raises(KeyError):
        m[4]
    out = m.lookup([1, 4, 5])
    assert out[0] == 10.0 and math.isnan(out[1]) and out[2] == 50.0
    assert np.isnan(ArrayMapping([], []).lookup([1])).all()
    with pytest.raises(ValueError):
        ArrayMapping([1, 2], [1.0])


def test_results_seg_to_row():
    assert results_seg_to_row(pd.Index([7, 3, 9])) == {7: 0, 3: 1, 9: 2}
    with pytest.raises(ValueError, match="duplicate"):
        results_seg_to_row([1, 1])


def test_network_mode():
    assert network_mode(SimpleNamespace(supernetwork_parameters={"columns": {"key": "flowline_id"}})) == "flowline"
    assert network_mode(SimpleNamespace(supernetwork_parameters={"columns": {"key": "flowpath_id"}})) == "flowpath"
    assert network_mode(SimpleNamespace(supernetwork_parameters=None)) == "flowpath"
