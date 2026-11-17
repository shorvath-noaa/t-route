"""Tree building, stop rules, drops, distance cap and theta resolution."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from troute.scaling_da import (
    DEFAULT_THETA,
    build_upstream_trees,
    trace_upstream_tree,
    cap_tree_by_distance,
    resolve_theta,
    tree_distance_km,
)

# Hand-built network (upstream -> downstream):
#
#   15 -> 14[gage B] -> 12 -\
#                            11 -> 10[gage A]
#   17 -> 16[lake] -\       /
#                    13 ---/
#            18 ----/
EDGES = {11: 10, 12: 11, 13: 11, 14: 12, 15: 14, 16: 13, 17: 16, 18: 13}
INCR = {10: 5.0, 11: 4.0, 12: 3.0, 13: 2.0, 14: 6.0, 15: 7.0, 16: 8.0, 17: 9.0, 18: 1.0}


@pytest.fixture
def net(g):
    return g.rconn_from_edges(EDGES), g.total_area(EDGES, INCR)


def members(tree):
    return set(tree.seg_ids.tolist())


# ---------------------------------------------------------------- single tree

def test_bfs_order_parents_and_junction_flags(net):
    rconn, area = net
    t = trace_upstream_tree(10, rconn, area, barrier_segments={14, 16}, theta=0.77, site="A")
    assert t.seg_ids.tolist() == [10, 11, 12, 13, 18]
    assert t.parent.tolist() == [-1, 0, 1, 1, 3]
    # 11's parent (10) has one upstream branch: linear. 12 and 13 hang off the
    # confluence at 11; 18 off the confluence at 13 (one sibling is the lake).
    assert t.is_junction_step.tolist() == [False, False, True, True, True]
    assert t.pruned_of(2).tolist() == [14]  # node 12 feeds from stopped gage B
    assert t.pruned_of(3).tolist() == [16]  # node 13 feeds from the lake
    assert t.gage_drainage_km2 == pytest.approx(area[10])
    assert np.all(t.parent[1:] < np.arange(1, t.n_nodes))


def test_headwater_gage_is_single_node(net):
    rconn, area = net
    t = trace_upstream_tree(15, rconn, area, barrier_segments=(), theta=0.5)
    assert t.seg_ids.tolist() == [15] and t.n_nodes == 1


def test_gage_in_stop_set_raises(net):
    rconn, area = net
    with pytest.raises(ValueError, match="barrier_segments"):
        trace_upstream_tree(10, rconn, area, barrier_segments={10}, theta=0.77)


def test_non_tree_connectivity_raises():
    # 3 listed upstream of both 1 and 2 (braid): not a tree.
    rconn = {0: [1, 2], 1: [3], 2: [3], 3: []}
    with pytest.raises(ValueError, match="not a tree"):
        trace_upstream_tree(0, rconn, {i: 1.0 for i in range(4)}, (), theta=0.77)


@pytest.mark.parametrize("theta", [0.0, -1.0, math.nan, math.inf])
def test_invalid_theta_rejected(net, theta):
    rconn, area = net
    with pytest.raises(ValueError, match="theta"):
        trace_upstream_tree(10, rconn, area, (), theta=theta)


# ---------------------------------------------------------------- many gages

def test_stops_at_upstream_gage_and_waterbody(net):
    rconn, area = net
    res = build_upstream_trees(
        rconn, {"A": 10, "B": 14}, area, lake_segments={16}
    )
    assert members(res.trees["A"]) == {10, 11, 12, 13, 18}
    assert members(res.trees["B"]) == {14, 15}
    assert not res.dropped


def test_trees_disjoint_and_upstream_gage_never_in_downstream_tree(net):
    rconn, area = net
    res = build_upstream_trees(rconn, {"A": 10, "B": 14}, area, lake_segments={16})
    a, b = members(res.trees["A"]), members(res.trees["B"])
    assert a.isdisjoint(b)


def test_waterbody_stop_can_be_disabled(net):
    rconn, area = net
    res = build_upstream_trees(
        rconn, {"A": 10}, area, lake_segments={16}, stop_at_lakes=False
    )
    assert {16, 17} <= members(res.trees["A"])


def test_holdout_gage_is_passed_through(net):
    # B held out: not a source and not a stop, so A's tree covers B's basin.
    rconn, area = net
    res = build_upstream_trees(rconn, {"A": 10}, area, lake_segments={16})
    assert {14, 15} <= members(res.trees["A"])
    assert "B" not in res.trees


def test_extra_stop_segs(net):
    rconn, area = net
    res = build_upstream_trees(rconn, {"A": 10}, area, extra_stop_segs={13})
    assert members(res.trees["A"]) == {10, 11, 12, 14, 15}
    t = res.trees["A"]
    assert t.pruned_of(1).tolist() == [13]


def test_colocated_gage_dropped_first_wins(net, caplog):
    rconn, area = net
    caplog.set_level(logging.WARNING)
    res = build_upstream_trees(rconn, {"A": 10, "A2": 10}, area)
    assert list(res.trees) == ["A"] and res.dropped == {"A2": "colocated"}
    assert "A2" in caplog.text


@pytest.mark.parametrize("bad", [math.nan, 0.0, -3.0])
def test_invalid_gage_area_drops_tree_and_logs(net, bad, caplog):
    rconn, area = net
    area = {**area, 10: bad}
    caplog.set_level(logging.WARNING)
    res = build_upstream_trees(rconn, {"A": 10, "B": 14}, area)
    assert "A" not in res.trees and res.dropped["A"] == "gage_area_invalid"
    assert "B" in res.trees
    assert "gage_area_invalid" not in caplog.text and "['A']" in caplog.text


def test_missing_gage_segment_area_drops_tree(net):
    rconn, area = net
    area = {k: v for k, v in area.items() if k != 10}
    res = build_upstream_trees(rconn, {"A": 10}, area)
    assert res.dropped == {"A": "gage_area_invalid"}


def test_nan_area_inside_tree_drops_tree(net, caplog):
    rconn, area = net
    area = {**area, 18: math.nan}
    caplog.set_level(logging.WARNING)
    res = build_upstream_trees(rconn, {"A": 10, "B": 14}, area, lake_segments={16})
    assert res.dropped == {"A": "nonfinite_area"} and "B" in res.trees
    assert "non-finite drainage area" in caplog.text


def test_nan_area_beyond_distance_cap_does_not_drop(net):
    rconn, area = net
    area = {**area, 15: math.nan}
    length = {s: 10.0 for s in INCR}
    res = build_upstream_trees(
        rconn, {"A": 10}, area, lake_segments={16}, length_km=length, max_reach_km=25.0
    )
    assert "A" in res.trees and 15 not in members(res.trees["A"])


def test_gage_on_waterbody_dropped(net):
    rconn, area = net
    res = build_upstream_trees(rconn, {"L": 16}, area, lake_segments={16})
    assert res.dropped == {"L": "gage_on_waterbody"}


def test_gage_area_override_used_as_A_o(net):
    rconn, area = net
    res = build_upstream_trees(rconn, {"A": 10}, area, gage_drainage_km2={"A": 123.0})
    t = res.trees["A"]
    assert t.gage_drainage_km2 == 123.0 and t.drainage_km2[1] == area[11]


def test_theta_by_site_applied_per_tree(net):
    rconn, area = net
    res = build_upstream_trees(
        rconn, {"A": 10, "B": 14}, area, site_theta={"B": 0.5}
    )
    assert res.trees["A"].theta == DEFAULT_THETA and res.trees["B"].theta == 0.5


def test_length_and_cap_must_come_together(net):
    rconn, area = net
    with pytest.raises(ValueError, match="together"):
        build_upstream_trees(rconn, {"A": 10}, area, max_reach_km=5.0)


# ---------------------------------------------------------------- distance cap

def test_distance_charges_parent_reaches(net):
    rconn, area = net
    t = trace_upstream_tree(10, rconn, area, barrier_segments={14, 16}, theta=0.77)
    L = {10: 1.0, 11: 2.0, 12: 4.0, 13: 8.0, 18: 16.0}
    # 11: L10; 12: L10+L11; 13: L10+L11; 18: L10+L11+L13
    assert tree_distance_km(t, L).tolist() == [0.0, 1.0, 3.0, 3.0, 11.0]


def test_cap_drops_far_nodes_and_keeps_them_in_denominator(net):
    rconn, area = net
    t = trace_upstream_tree(10, rconn, area, barrier_segments={14, 16}, theta=0.77)
    L = {10: 1.0, 11: 2.0, 12: 4.0, 13: 8.0, 18: 16.0}
    capped, n = cap_tree_by_distance(t, L, 10.0)
    assert n == 1 and capped.seg_ids.tolist() == [10, 11, 12, 13]
    assert capped.parent.tolist() == [-1, 0, 1, 1]
    # 18 is now a pruned neighbour of 13, next to the lake 16.
    assert sorted(capped.pruned_of(3).tolist()) == [16, 18]
    assert capped.is_junction_step.tolist() == [False, False, True, True]


def test_cap_boundary_inclusive(net):
    rconn, area = net
    t = trace_upstream_tree(10, rconn, area, barrier_segments={14, 16}, theta=0.77)
    L = {10: 1.0, 11: 2.0, 12: 4.0, 13: 8.0, 18: 16.0}
    capped, n = cap_tree_by_distance(t, L, 11.0)
    assert n == 0 and capped is t


@pytest.mark.parametrize("bad", [None, math.nan, 0.0, -1.0])
def test_unusable_length_fails_closed(net, bad):
    rconn, area = net
    t = trace_upstream_tree(10, rconn, area, barrier_segments={14, 16}, theta=0.77)
    L = {10: 1.0, 11: 2.0, 12: 4.0, 18: 16.0}
    if bad is not None:
        L[13] = bad
    capped, n = cap_tree_by_distance(t, L, 1000.0)
    assert 18 not in members(capped) and 13 in members(capped) and n == 1


@pytest.mark.parametrize("cap", [0.0, -5.0, math.inf, math.nan])
def test_invalid_cap_rejected(net, cap):
    rconn, area = net
    t = trace_upstream_tree(10, rconn, area, barrier_segments=(), theta=0.77)
    with pytest.raises(ValueError):
        cap_tree_by_distance(t, {}, cap)


# ---------------------------------------------------------------- theta

def test_theta_precedence_site_over_vpu_over_default():
    r = resolve_theta(
        ["s1", "s2", "s3", "s4"],
        default=0.7,
        by_vpu={"01": 0.6, "03N": 0.4},
        per_site={"s1": 0.9},
        site_vpu={"s1": "01", "s2": "01", "s3": "02", "s4": "03N"},
    )
    assert r.site_theta == {"s1": 0.9, "s2": 0.6, "s3": 0.7, "s4": 0.4}
    assert r.source_by_site == {"s1": "site", "s2": "vpu", "s3": "default", "s4": "vpu"}


def test_theta_default_and_missing_vpu():
    r = resolve_theta(["a", "b"], site_vpu={"a": "05"})
    assert r.site_theta == {"a": DEFAULT_THETA, "b": DEFAULT_THETA}


def test_theta_unused_keys_reported():
    r = resolve_theta(
        ["s1"], by_vpu={"01": 0.6, "17": 0.5}, per_site={"nope": 0.3}, site_vpu={"s1": "01"}
    )
    assert r.unused_vpu_keys == {"17"} and r.unused_site_keys == {"nope"}


def test_theta_vpu_keys_are_strings_with_leading_zeros():
    r = resolve_theta(["s"], by_vpu={"01": 0.6}, site_vpu={"s": "01"})
    assert r.source_by_site["s"] == "vpu"
    r = resolve_theta(["s"], by_vpu={"1": 0.6}, site_vpu={"s": "01"})
    assert r.source_by_site["s"] == "default" and r.unused_vpu_keys == {"1"}


@pytest.mark.parametrize("kw", [
    {"default": 0.0}, {"default": math.nan}, {"by_vpu": {"01": -1}},
    {"per_site": {"s": math.inf}}, {"per_site": {"s": "abc"}},
])
def test_theta_invalid_values_raise(kw):
    with pytest.raises(ValueError):
        resolve_theta(["s"], **kw)
