"""Spread of gage increments over upstream trees: hand cases and a seeded oracle."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from troute.scaling_da import (
    UpstreamTree,
    apply_increments,
    apply_upstream_correction,
    build_upstream_trees,
    trace_upstream_tree,
    dq_o_matrix,
    pack_trees,
    spread_increments,
)

TH = 0.77


def rows_identity(segs):
    """seg id -> row, with rows deliberately not equal to ids."""
    return {s: i for i, s in enumerate(sorted(segs))}


def one_tree(rconn, area, gage, stop=(), theta=TH, site="G"):
    return {site: trace_upstream_tree(gage, rconn, area, stop, theta=theta, site=site)}


def flows(seg_to_row, values, nt=1, dtype=np.float32):
    q = np.zeros((len(seg_to_row), nt), dtype=dtype)
    for s, v in values.items():
        q[seg_to_row[s]] = v
    return q


def inc_by_seg(packed, inc):
    """seg-agnostic: map row -> increment row."""
    return {int(r): inc[k] for k, r in enumerate(packed.rows)}


# ---------------------------------------------------------------- linear chain

def test_linear_chain_matches_closed_form():
    # 3 -> 2 -> 1 -> 0[gage]
    rconn = {0: [1], 1: [2], 2: [3], 3: []}
    area = {0: 100.0, 1: 60.0, 2: 30.0, 3: 10.0}
    s2r = rows_identity(area)
    nt = 5
    q = np.tile(np.array([[50.0], [30.0], [15.0], [5.0]], dtype=np.float32), (1, nt))
    dq = np.array([[2.0, -1.0, 0.0, 4.5, -7.25]])
    packed, dropped = pack_trees(one_tree(rconn, area, 0), s2r)
    assert not dropped
    inc = spread_increments(packed, q, dq)
    for k, r in enumerate(packed.rows):
        seg = [s for s, rr in s2r.items() if rr == r][0]
        expected = 0.0 if seg == 0 else dq[0] * (area[seg] / area[0]) ** TH
        np.testing.assert_allclose(inc[k], expected, rtol=1e-13, atol=0)


def test_root_untouched_by_default_and_included_on_request():
    rconn = {0: [1], 1: []}
    area = {0: 10.0, 1: 5.0}
    s2r = rows_identity(area)
    q = flows(s2r, {0: 4.0, 1: 2.0})
    packed, _ = pack_trees(one_tree(rconn, area, 0), s2r)
    dq = np.array([[3.0]])
    assert spread_increments(packed, q, dq)[0, 0] == 0.0
    assert spread_increments(packed, q, dq, include_root=True)[0, 0] == 3.0


def test_theta_changes_linear_factor():
    rconn = {0: [1], 1: []}
    area = {0: 16.0, 1: 4.0}
    s2r = rows_identity(area)
    q = flows(s2r, {0: 1.0, 1: 1.0})
    packed, _ = pack_trees(one_tree(rconn, area, 0, theta=0.5), s2r)
    inc = spread_increments(packed, q, np.array([[8.0]]))
    assert inc[1, 0] == pytest.approx(8.0 * 0.5)


def test_area_inversion_clamped_to_one_and_counted(caplog):
    rconn = {0: [1], 1: []}
    area = {0: 10.0, 1: 40.0}  # area grows upstream: hydrofabric defect
    s2r = rows_identity(area)
    q = flows(s2r, {0: 1.0, 1: 1.0})
    caplog.set_level(logging.WARNING)
    packed, _ = pack_trees(one_tree(rconn, area, 0), s2r)
    assert packed.n_area_inversions == 1 and "clamped" in caplog.text
    assert spread_increments(packed, q, np.array([[2.0]]))[1, 0] == 2.0


# ---------------------------------------------------------------- confluences
#
#   4 -> 2 -\
#            1 -> 0[gage]
#        3 -/

CONF = {0: [1], 1: [2, 3], 2: [4], 3: [], 4: []}
CONF_AREA = {0: 50.0, 1: 40.0, 2: 25.0, 3: 12.0, 4: 9.0}


def conf_setup(stop=(), values=None, theta=TH):
    s2r = rows_identity(CONF_AREA)
    values = values or {0: 11.0, 1: 10.0, 2: 6.0, 3: 3.0, 4: 2.0}
    q = flows(s2r, values)
    packed, _ = pack_trees(one_tree(CONF, CONF_AREA, 0, stop=stop, theta=theta), s2r)
    return s2r, q, packed


def mult_of(packed, inc, s2r, seg, dq):
    return inc_by_seg(packed, inc)[s2r[seg]][0] / dq


def test_confluence_split_by_flow_when_parent_exceeds_branch_sum():
    s2r, q, packed = conf_setup()
    dq = 4.0
    inc = spread_increments(packed, q, np.array([[dq]]))
    m1 = (40.0 / 50.0) ** TH
    assert mult_of(packed, inc, s2r, 1, dq) == pytest.approx(m1)
    assert mult_of(packed, inc, s2r, 2, dq) == pytest.approx(m1 * 6.0 / 10.0)
    assert mult_of(packed, inc, s2r, 3, dq) == pytest.approx(m1 * 3.0 / 10.0)
    # above a confluence the area step restarts from the branch, not the gage
    assert mult_of(packed, inc, s2r, 4, dq) == pytest.approx(m1 * 0.6 * (9.0 / 25.0) ** TH)


def test_confluence_denominator_clamped_to_branch_sum():
    # routing lag: parent flow below the sum of its branches -> shares sum to 1
    s2r, q, packed = conf_setup(values={0: 11.0, 1: 7.0, 2: 6.0, 3: 3.0, 4: 2.0})
    inc = spread_increments(packed, q, np.array([[1.0]]))
    m1 = mult_of(packed, inc, s2r, 1, 1.0)
    s2 = mult_of(packed, inc, s2r, 2, 1.0) / m1
    s3 = mult_of(packed, inc, s2r, 3, 1.0) / m1
    assert s2 == pytest.approx(6.0 / 9.0) and s3 == pytest.approx(3.0 / 9.0)
    assert s2 + s3 <= 1.0 + 1e-12


def test_pruned_sibling_counts_in_denominator_and_keeps_its_share():
    # branch 3 is stopped (another gage): it receives nothing, but its flow
    # stays in the denominator so branch 2 gets only its own share.
    s2r, q, packed = conf_setup(stop={3}, values={0: 11.0, 1: 7.0, 2: 6.0, 3: 3.0, 4: 2.0})
    assert s2r[3] not in packed.rows.tolist()
    inc = spread_increments(packed, q, np.array([[1.0]]))
    m1 = mult_of(packed, inc, s2r, 1, 1.0)
    share2 = mult_of(packed, inc, s2r, 2, 1.0) / m1
    assert share2 == pytest.approx(6.0 / 9.0)  # not 6/7
    assert share2 < 1.0


def test_min_flow_floor_zeroes_branch_and_its_subtree():
    s2r, q, packed = conf_setup(values={0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0})
    inc = spread_increments(packed, q, np.array([[5.0]]))
    by = inc_by_seg(packed, inc)
    assert by[s2r[1]][0] != 0.0
    assert by[s2r[2]][0] == 0.0 and by[s2r[3]][0] == 0.0 and by[s2r[4]][0] == 0.0
    # a floor above the denominator also zeroes the split
    s2r, q, packed = conf_setup()
    inc = spread_increments(packed, q, np.array([[5.0]]), split_floor_cms=50.0)
    assert inc_by_seg(packed, inc)[s2r[2]][0] == 0.0


@pytest.mark.parametrize("nan_seg", [1, 2])
def test_nan_flow_gives_zero_not_nan(nan_seg):
    vals = {0: 11.0, 1: 10.0, 2: 6.0, 3: 3.0, 4: 2.0}
    vals[nan_seg] = math.nan
    s2r, q, packed = conf_setup(values=vals)
    inc = spread_increments(packed, q, np.array([[5.0]]))
    assert np.all(np.isfinite(inc))
    assert inc_by_seg(packed, inc)[s2r[2]][0] == 0.0


def test_root_flow_override_changes_root_split():
    # gage with two upstream branches; published gage flow is the nudged value
    rconn = {0: [1, 2], 1: [], 2: []}
    area = {0: 10.0, 1: 6.0, 2: 3.0}
    s2r = rows_identity(area)
    q = flows(s2r, {0: 30.0, 1: 6.0, 2: 3.0})  # nudged up from background 10
    packed, _ = pack_trees(one_tree(rconn, area, 0), s2r)
    dq = np.array([[20.0]])
    a = spread_increments(packed, q, dq)
    b = spread_increments(packed, q, dq, root_flow=np.array([[10.0]]))
    assert a[1, 0] == pytest.approx(20.0 * 6.0 / 30.0)
    assert b[1, 0] == pytest.approx(20.0 * 6.0 / 10.0)


# ---------------------------------------------------------------- write-back

def test_apply_floors_at_zero_and_reports_applied():
    rconn = {0: [1], 1: []}
    area = {0: 10.0, 1: 10.0}
    s2r = rows_identity(area)
    q = flows(s2r, {0: 5.0, 1: 2.0})
    packed, _ = pack_trees(one_tree(rconn, area, 0), s2r)
    res = apply_upstream_correction(q, packed, np.array([[-5.0]]), dt_s=300.0, return_applied=True)
    assert q[s2r[1], 0] == 0.0
    assert res.applied[1, 0] == pytest.approx(-2.0)
    assert res.injected_volume_m3 == pytest.approx(-600.0)
    assert res.n_segments_changed == 1


def test_segments_outside_trees_are_bit_identical(g):
    rconn = {0: [1, 9], 1: [], 9: [8], 8: []}
    area = {0: 20.0, 1: 8.0, 9: 9.0, 8: 4.0}
    s2r = rows_identity(area)
    q = np.random.default_rng(1).uniform(0.1, 10, (4, 6)).astype(np.float32)
    before = q.copy()
    packed, _ = pack_trees(one_tree(rconn, area, 0, stop={9}), s2r)
    apply_upstream_correction(q, packed, np.full((1, 6), 2.5))
    for s in (0, 9, 8):  # root, stopped branch, and above it
        assert q[s2r[s]].tobytes() == before[s2r[s]].tobytes()
    assert not np.array_equal(q[s2r[1]], before[s2r[1]])
    assert q.dtype == np.float32


def test_zero_innovation_writes_nothing():
    s2r, q, packed = conf_setup()
    before = q.copy()
    res = apply_upstream_correction(q, packed, np.zeros((1, 1)))
    assert q.tobytes() == before.tobytes() and res.n_segments_changed == 0


def test_float32_writeback_tolerance():
    s2r, q, packed = conf_setup()
    q0 = q.astype(np.float64)
    dq = np.array([[0.123456789]])
    inc = spread_increments(packed, q, dq)
    applied = apply_increments(q, packed.rows, inc)
    # float32 storage: applied differs from the float64 increment by <= 1 ulp of q
    tol = np.spacing(np.abs(q0[packed.rows]).astype(np.float32)).astype(np.float64)
    assert np.all(np.abs(applied - inc) <= tol + 1e-15)


# ---------------------------------------------------------------- inputs

def test_nonfinite_innovation_treated_as_zero(caplog):
    s2r, q, packed = conf_setup()
    caplog.set_level(logging.WARNING)
    inc = spread_increments(packed, np.repeat(q, 2, 1), np.array([[math.nan, 1.0]]))
    assert np.all(inc[:, 0] == 0.0) and np.any(inc[:, 1] != 0.0)
    assert "non-finite" in caplog.text


def test_shape_checks():
    s2r, q, packed = conf_setup()
    with pytest.raises(ValueError, match="dq_o shape"):
        spread_increments(packed, q, np.zeros((2, 1)))
    with pytest.raises(ValueError, match="root_flow"):
        spread_increments(packed, q, np.zeros((1, 1)), root_flow=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="2-D"):
        spread_increments(packed, q[:, 0], np.zeros((1, 1)))


def test_pack_drops_tree_with_missing_member_or_pruned_row():
    s2r = rows_identity(CONF_AREA)
    trees = one_tree(CONF, CONF_AREA, 0, stop={3})
    no_member = {k: v for k, v in s2r.items() if k != 4}
    packed, dropped = pack_trees(trees, no_member)
    assert packed.n_trees == 0 and "G" in dropped
    no_pruned = {k: v for k, v in s2r.items() if k != 3}
    packed, dropped = pack_trees(trees, no_pruned)
    assert packed.n_trees == 0 and "G" in dropped


def test_pack_rejects_overlapping_trees():
    t1 = trace_upstream_tree(0, CONF, CONF_AREA, (), theta=TH)
    t2 = trace_upstream_tree(1, CONF, CONF_AREA, (), theta=TH)
    with pytest.raises(ValueError, match="disjoint"):
        pack_trees({"a": t1, "b": t2}, rows_identity(CONF_AREA))


def test_dq_o_matrix_orders_by_site_and_fills_zeros():
    t = {s: trace_upstream_tree(g, {g: []}, {g: 1.0}, (), theta=TH, site=s)
         for s, g in (("x", 5), ("y", 6))}
    packed, _ = pack_trees(t, {5: 0, 6: 1})
    m = dq_o_matrix(packed, {"y": np.array([1.0, 2.0])}, 2)
    assert m.tolist() == [[0.0, 0.0], [1.0, 2.0]]
    with pytest.raises(ValueError):
        dq_o_matrix(packed, {"x": np.zeros(3)}, 2)


def test_empty_pack():
    packed, _ = pack_trees({}, {})
    assert spread_increments(packed, np.zeros((3, 4)), np.zeros((0, 4))).shape == (0, 4)


# ---------------------------------------------------------------- seeded oracle

def oracle(trees, rconn, area, q, s2r, dq_by_site, min_flow=1e-6):
    """Node-by-node reference, independent of the packed layout.

    Confluence denominators come straight from the full reverse connectivity
    (not from the trees' pruned lists), which also checks the pruned/cap
    bookkeeping.
    """
    nt = q.shape[1]
    out = {}
    for gage_id, t in trees.items():
        dq = dq_by_site[gage_id]
        mult = {}
        for i, seg in enumerate(t.seg_ids.tolist()):
            m = np.empty(nt)
            for k in range(nt):
                if i == 0:
                    m[k] = 1.0
                    continue
                p = int(t.parent[i])
                pseg = int(t.seg_ids[p])
                if t.is_junction_step[i]:
                    qp = float(q[s2r[pseg], k])
                    bs = sum(float(q[s2r[b], k]) for b in rconn[pseg])
                    denom = max(qp, bs)
                    f = float(q[s2r[seg], k]) / denom if denom > min_flow else 0.0
                else:
                    a_p = t.drainage_km2[p]
                    f = (t.drainage_km2[i] / a_p) ** t.theta if a_p > 0 else 0.0
                f = min(max(f, 0.0), 1.0)
                m[k] = mult[p][k] * f
            mult[i] = m
            out[seg] = 0.0 * dq if i == 0 else m * dq
    return out


@pytest.mark.parametrize("seed", range(8))
def test_random_trees_against_oracle_and_invariants(g, seed):
    rng = np.random.default_rng(1000 + seed)
    n, nt = 300, 7
    edges, area = g.random_network(rng, n)
    rconn = g.rconn_from_edges(edges)
    segs = sorted(area)
    gages = rng.choice(segs, size=12, replace=False).tolist()
    lakes = set(rng.choice([s for s in segs if s not in gages], size=6, replace=False).tolist())
    gage_segments = {f"g{s}": s for s in gages}
    length = {s: float(rng.uniform(0.2, 5.0)) for s in segs}
    res = build_upstream_trees(
        rconn, gage_segments, area,
        site_theta={site: float(rng.uniform(0.4, 1.0)) for site in gage_segments},
        lake_segments=lakes, length_km=length, max_reach_km=float(rng.uniform(5, 40)),
    )
    s2r = {s: i for i, s in enumerate(rng.permutation(segs).tolist())}
    q = rng.uniform(0.0, 50.0, (n, nt))
    q[rng.random((n, nt)) < 0.05] = 0.0  # some dry cells
    q = q.astype(np.float32)
    dq_by_site = {s: rng.normal(0, 10, nt) for s in res.trees}
    packed, dropped = pack_trees(res.trees, s2r)
    assert not dropped
    dq = dq_o_matrix(packed, dq_by_site, nt)

    inc = spread_increments(packed, q, dq)
    inc_chunked = spread_increments(packed, q, dq, chunk_timesteps=3)
    assert np.array_equal(inc, inc_chunked)

    ref = oracle(res.trees, rconn, area, q, s2r, dq_by_site)
    r2k = {int(r): k for k, r in enumerate(packed.rows)}
    for seg, expected in ref.items():
        np.testing.assert_allclose(inc[r2k[s2r[seg]]], expected, rtol=1e-12, atol=1e-12)

    # |M| <= 1 everywhere: no node gets more than its gage's increment
    gage_dq = dq[packed.tree_of]
    assert np.all(np.abs(inc) <= np.abs(gage_dq) * (1 + 1e-12))

    # confluence shares sum to <= 1 (relative to the parent's multiplier)
    for t in res.trees.values():
        for p in range(t.n_nodes):
            kids = np.flatnonzero((t.parent == p) & t.is_junction_step)
            if kids.size == 0:
                continue
            for k in range(nt):
                qp = float(q[s2r[int(t.seg_ids[p])], k])
                bs = sum(float(q[s2r[b], k]) for b in rconn[int(t.seg_ids[p])])
                denom = max(qp, bs)
                if denom > 1e-6:
                    tot = sum(float(q[s2r[int(t.seg_ids[c])], k]) for c in kids) / denom
                    assert tot <= 1.0 + 1e-12

    # closed form on nodes whose path to the gage has no confluence step
    for gage_id, t in res.trees.items():
        clean = np.zeros(t.n_nodes, dtype=bool)
        clean[0] = True
        for i in range(1, t.n_nodes):
            clean[i] = clean[t.parent[i]] and not t.is_junction_step[i]
        for i in np.flatnonzero(clean)[1:]:
            seg = int(t.seg_ids[i])
            expected = dq_by_site[gage_id] * min(1.0, (t.drainage_km2[i] / t.drainage_km2[0]) ** t.theta)
            np.testing.assert_allclose(inc[r2k[s2r[seg]]], expected, rtol=1e-12)

    # fused chunked write-back == materialize-then-apply, bit for bit
    q_fused = q.copy()
    res_f = apply_upstream_correction(q_fused, packed, dq, chunk_timesteps=2, dt_s=60.0, return_applied=True)
    q_before = q.copy()
    applied = apply_increments(q, packed.rows, inc)
    assert q_fused.tobytes() == q.tobytes()
    assert np.array_equal(res_f.applied, applied)
    assert res_f.injected_volume_m3 == pytest.approx(applied.sum() * 60.0, rel=1e-12)
    assert np.array_equal(res_f.changed, np.any(applied != 0, axis=1))
    q = q_before.copy()
    # write-back touches tree members only
    apply_increments(q, packed.rows, inc)
    outside = np.setdiff1d(np.arange(n), packed.rows)
    assert q[outside].tobytes() == q_before[outside].tobytes()
    assert np.all(q[packed.rows] >= 0.0)

    # stop rules: no tree contains a lake or another gage's segment
    members = np.concatenate([t.seg_ids for t in res.trees.values()])
    assert lakes.isdisjoint(members.tolist())
    for gage_id, t in res.trees.items():
        others = set(gages) - {t.gage_segment}
        assert others.isdisjoint(t.seg_ids.tolist())


def test_negative_branch_flow_gives_zero_share():
    # Defensive bound: a negative modeled flow (input defect) must not flip the
    # sign of the correction.
    s2r, q, packed = conf_setup(values={0: 11.0, 1: 10.0, 2: -2.0, 3: 3.0, 4: 2.0})
    inc = spread_increments(packed, q, np.array([[5.0]]))
    by = inc_by_seg(packed, inc)
    assert by[s2r[2]][0] == 0.0 and by[s2r[4]][0] == 0.0
    assert by[s2r[3]][0] > 0.0
