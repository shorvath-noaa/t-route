"""Run-time driver of area-scaled streamflow DA (troute.scaling_da.runtime), no I/O.

Routed graph used throughout (upstream -> downstream); lake 900 replaces its
member segments 901 and 902; gages A, B, C (nwis) and R (rfc):

    8 -> 7 -> 6[C] -> 5 -> 4[B] --\\
                                   3 -> 2 -> 1[A]
    11 -> (lake 900) -> 10 -> 9 --/
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from troute.scaling_da import pack_trees, spread_increments
from troute.scaling_da.runtime import (
    DEPTH_EXPONENT,
    DEPTH_RATIO_CAP,
    ScalingDAOptions,
    ScalingDARuntime,
    observed_sites,
    read_site_list,
    read_theta_csv,
    run_timeslice_files,
    scaling_enabled,
)

EDGES = {2: 1, 3: 2, 4: 3, 9: 3, 5: 4, 6: 5, 7: 6, 8: 7, 10: 9, 900: 10, 11: 900}
INCR = {1: 2.0, 2: 3.0, 3: 1.0, 4: 4.0, 5: 5.0, 6: 2.0, 7: 6.0, 8: 8.0, 9: 1.5, 10: 2.5, 900: 0.0, 11: 7.0}
LAKE_MEMBERS = {901: 900, 902: 900}
GAGE_SEG = {"A": 1, "B": 4, "C": 6}
ALL_SEGS = sorted(set(EDGES) | set(EDGES.values()))


def total_area():
    up = {}
    for u, d in EDGES.items():
        up.setdefault(d, []).append(u)
    tot = {}

    def acc(s):
        if s not in tot:
            tot[s] = INCR[s] + sum(acc(u) for u in up.get(s, ()))
        return tot[s]

    for s in ALL_SEGS:
        acc(s)
    return tot


def make_network(*, vpu="01"):
    tot = total_area()
    rconn = {s: [] for s in ALL_SEGS}
    for u, d in sorted(EDGES.items()):
        rconn[d].append(u)
    segs = [s for s in ALL_SEGS if s != 900] + list(LAKE_MEMBERS)
    area = {**tot, **{m: 20.0 for m in LAKE_MEMBERS}}
    seg_attrs = pd.DataFrame(
        {"total_dasqkm": [area[s] for s in segs], "vpuid": [vpu] * len(segs)}, index=pd.Index(segs, name="key")
    )
    dataframe = pd.DataFrame({"dx": [1000.0] * len(segs)}, index=pd.Index(segs, name="key"))
    rows = [("fp-1", "nwis-A"), ("fp-4", "nwis-B"), ("fp-6", "nwis-C"), ("fp-3", "rfc-R")]
    hl = pd.DataFrame({"flowpath_id": [r[0] for r in rows], "hl_reference": [r[1] for r in rows],
                       "vpuid": [vpu] * len(rows)})
    parts = hl["hl_reference"].str.partition("-")
    hl["type"], hl["site"] = parts[0], parts[2]
    return SimpleNamespace(
        reverse_network=rconn,
        dataframe=dataframe,
        segment_attributes=seg_attrs,
        gage_hydrolocations=hl,
        gages={"gages": {1: "A", 4: "B", 6: "C", 3: "R"}},
        waterbody_connections=dict(LAKE_MEMBERS),
        compute_parameters={"compute_nodes": False},
        supernetwork_parameters={"columns": {"key": "flowpath_id"}},
        diffusive_network_data={},
    )


def da_params(tmp_path=None, **params):
    streamflow = dict(streamflow_nudging=True, streamflow_scaling=True, streamflow_scaling_parameters=params)
    folder = str(tmp_path) if tmp_path else "/tmp"
    return dict(usgs_timeslices_folder=folder, qc_threshold=1, streamflow_da=streamflow)


def build(observed=("A", "B", "C"), net=None, **params):
    return ScalingDARuntime.build(net or make_network(), da_params(**{}, **params), observed=observed)


# ------------------------------------------------------------------ switches


def test_off_returns_none():
    off = da_params()
    off["streamflow_da"].update(streamflow_scaling=False)
    assert ScalingDARuntime.build(make_network(), off, observed=["A"]) is None
    assert ScalingDARuntime.build(make_network(), {}, observed=["A"]) is None
    assert not scaling_enabled({"streamflow_da": None})


def test_requires_nudging():
    p = da_params()
    p["streamflow_da"]["streamflow_nudging"] = False
    with pytest.raises(ValueError, match="streamflow_nudging"):
        ScalingDARuntime.build(make_network(), p, observed=["A"])


def test_hyfeatures_only():
    net = make_network()
    del net.segment_attributes
    with pytest.raises(NotImplementedError, match="HYFeatures"):
        ScalingDARuntime.build(net, da_params(), observed=["A"])


def test_options_defaults_and_chunk():
    o = ScalingDAOptions.from_parameters({})
    assert (o.default_theta, o.max_reach_km, o.split_floor_cms, o.site_types, o.chunk_timesteps) == (
        0.77, 200.0, 1e-6, ("nwis",), 24)
    assert ScalingDAOptions.from_parameters({"spread_chunk_timesteps": 0}).chunk_timesteps is None
    assert ScalingDAOptions.from_parameters({"spread_chunk_timesteps": 6}).chunk_timesteps == 6


# ------------------------------------------------------------------ sources


def members(rt, site):
    return sorted(int(s) for s in rt.trees[site].seg_ids)


def test_all_observed_sources_and_trees():
    rt = build()
    assert rt.sources == {"A", "B", "C"} and not rt.held_out and not rt.no_observations
    assert members(rt, "A") == [1, 2, 3, 9, 10]  # stops at B (4) and the lake (900)
    assert members(rt, "B") == [4, 5]
    assert members(rt, "C") == [6, 7, 8]
    assert set(rt.trees["A"].pruned_ids.tolist()) == {4, 900}
    rep = rt.report.set_index("site")
    assert rep.loc[["A", "B", "C"], "status"].tolist() == ["source"] * 3
    assert "R" not in rep.index  # other types are not assimilated


def test_gage_without_observations_is_not_a_stop():
    rt = build(observed=("A", "B"))
    assert rt.no_observations == {"C"} and "C" not in rt.trees
    assert members(rt, "B") == [4, 5, 6, 7, 8]
    assert rt.report.set_index("site").loc["C", "status"] == "no_observations"


def test_observed_gage_of_other_type_is_a_stop():
    """R (rfc, at 3) is nudged when observed, so trees must not cross it."""
    rt = build(observed=("A", "B", "C", "R"))
    assert "R" not in rt.trees
    assert members(rt, "A") == [1, 2]
    assert 3 in set(rt.trees["A"].pruned_ids.tolist())
    assert members(build(), "A") == [1, 2, 3, 9, 10]  # unobserved: not a stop


def test_holdout_removes_gage_from_nudging_and_trees(tmp_path):
    listing = tmp_path / "eval_sites.txt"
    listing.write_text("# evaluation gages\nB\n\n")
    rt = build(holdout_sites_file=str(listing))
    assert rt.held_out == {"B"} and rt.held_out_segs == {4}
    assert "B" not in rt.trees
    assert members(rt, "A") == [1, 2, 3, 4, 5, 9, 10]  # still stops at C
    usgs = pd.DataFrame({"t0": [1.0, 2.0, 3.0]}, index=[1, 4, 6])
    lastobs = pd.DataFrame({"lastobs_discharge": [1.0, 2.0]}, index=[4, 6])
    u, lo = rt.nudging_inputs(usgs, lastobs)
    assert u.index.tolist() == [1, 6] and lo.index.tolist() == [6]
    assert rt.report.set_index("site").loc["B", "status"] == "held_out"


def test_unknown_holdout_id_raises(tmp_path):
    listing = tmp_path / "eval_sites.txt"
    listing.write_text("B\nZZZ\n")
    with pytest.raises(ValueError, match="ZZZ"):
        build(holdout_sites_file=str(listing))


def test_holdout_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        build(holdout_sites_file=str(tmp_path / "nope.txt"))


def test_no_holdout_leaves_nudging_inputs_alone():
    rt = build()
    usgs = pd.DataFrame({"t0": [1.0]}, index=[4])
    u, lo = rt.nudging_inputs(usgs, pd.DataFrame())
    assert u is usgs


def test_no_observed_gage_in_domain_warns(caplog):
    caplog.set_level(logging.WARNING)
    rt = build(observed=("Q",))
    assert not rt.trees and "none of the" in caplog.text


def test_empty_roster_from_files_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no usable observations"):
        ScalingDARuntime.build(make_network(), da_params(tmp_path), da_sets=[{}])


# ------------------------------------------------------------------ theta


def test_theta_precedence_csv_vpu_default(tmp_path):
    csv = tmp_path / "theta.csv"
    csv.write_text("gage_id,theta\nC,0.5\n")
    rt = build(theta={"default": 0.9, "by_vpu": {"01": 0.6}, "per_tree_csv": str(csv)})
    assert rt.trees["C"].theta == 0.5
    assert rt.trees["A"].theta == 0.6 and rt.trees["B"].theta == 0.6
    rep = rt.report.set_index("site")
    assert rep.loc["C", "theta_source"] == "site" and rep.loc["A", "theta_source"] == "vpu"


def test_theta_keys_matching_nothing_are_errors(tmp_path):
    with pytest.raises(ValueError, match="by_vpu"):
        build(theta={"by_vpu": {"1": 0.6}})  # not zero-padded
    csv = tmp_path / "theta.csv"
    csv.write_text("1104000,0.6\n")
    with pytest.raises(ValueError, match="per_tree_csv"):
        build(theta={"per_tree_csv": str(csv)})


def test_theta_keys_partly_unused_warn(tmp_path, caplog):
    csv = tmp_path / "theta.csv"
    csv.write_text("A,0.6\nOTHER,0.7\n")
    caplog.set_level(logging.WARNING)
    rt = build(theta={"by_vpu": {"01": 0.65, "02": 0.5}, "per_tree_csv": str(csv)})
    assert rt.trees["A"].theta == 0.6 and rt.trees["B"].theta == 0.65
    assert "OTHER" in caplog.text and "'02'" in caplog.text


def test_read_theta_csv(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("# comment\ngage_id,theta\n01104000, 0.6\n\n01105000,0.7 # note\n")
    assert read_theta_csv(f) == {"01104000": 0.6, "01105000": 0.7}
    f.write_text("01104000,0.6\n")  # no header
    assert read_theta_csv(f) == {"01104000": 0.6}
    for text, match in [("a,0.6\na,0.7\n", "duplicate"), ("a,0.6\nb,x\n", "number"), ("a,0\n", "> 0"),
                        ("a\n", "expected"), ("gage_id,theta\n", "no rows"), ("a,nan\n", "finite")]:
        f.write_text(text)
        with pytest.raises(ValueError, match=match):
            read_theta_csv(f)
    with pytest.raises(FileNotFoundError):
        read_theta_csv(tmp_path / "missing.csv")


def test_read_site_list(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("01104000\n  01105000  \n# x\n\n01209500 # trailing\n")
    assert read_site_list(f) == ["01104000", "01105000", "01209500"]


# ------------------------------------------------------------------ roster


def fake_reader(tables):
    def read(path):
        obs, qual = tables[str(path)]
        return obs, qual

    return read


def frames(rows):
    """rows: station -> list of (obs, quality fraction)."""
    cols = [f"2019-10-28_00:{15 * i:02d}:00" for i in range(max(len(v) for v in rows.values()))]
    obs = pd.DataFrame({c: [v[i][0] if i < len(v) else np.nan for v in rows.values()] for i, c in enumerate(cols)},
                       index=list(rows))
    qual = pd.DataFrame({c: [v[i][1] if i < len(v) else np.nan for v in rows.values()] for i, c in enumerate(cols)},
                        index=list(rows))
    return obs, qual


def test_observed_sites_screening():
    tables = {
        "f1": frames({"A": [(5.0, 1.0)], "B": [(np.nan, 1.0)], "C": [(3.0, 0.5)], "D": [(0.0, 1.0)]}),
        "f2": frames({"B": [(np.nan, 1.0), (2.0, 1.0)], "E": [(4.0, 1.5)]}),
        "f3": (pd.DataFrame(), pd.DataFrame()),
    }
    got = observed_sites(["f1", "f2", "f3"], qc_threshold=1.0, reader=fake_reader(tables))
    assert got == {"A", "B"}  # C fails qc, D has q <= 0, E has an invalid quality flag
    assert observed_sites(["f1"], qc_threshold=0.5, reader=fake_reader(tables)) == {"A", "C"}
    assert observed_sites([], reader=fake_reader(tables)) == set()


def test_run_timeslice_files_union(tmp_path):
    sets = [{"usgs_timeslice_files": ["b.ncdf", "a.ncdf"]}, {"usgs_timeslice_files": ["b.ncdf", "c.ncdf"]}, {}]
    files = run_timeslice_files({"usgs_timeslices_folder": str(tmp_path)}, sets)
    assert [f.name for f in files] == ["a.ncdf", "b.ncdf", "c.ncdf"]
    assert run_timeslice_files({}, sets) == []


# ------------------------------------------------------------------ apply


NTS = 6


def model_flows(rng):
    """Pre-nudge flow per routed node and timestep (lake node included)."""
    return {s: rng.uniform(1.0, 50.0, NTS).astype(np.float32) for s in ALL_SEGS}


def make_results(qm, nudges, *, depth=None, split=((1, 2, 3, 9, 10, 11, 900), (4, 5, 6, 7, 8)), rng=None):
    """Fake routing results: published flow = model flow + nudge at gages.

    `nudges`: gage segment -> (nts,) correction; column 0 of the kernel array is set
    to garbage to check that it is ignored.
    """
    rng = rng or np.random.default_rng(0)
    out = []
    for part in split:
        ids = np.asarray(rng.permutation(list(part)), dtype=np.int64)
        fvd = np.zeros((ids.size, NTS * 3), dtype=np.float32)
        for i, s in enumerate(ids):
            q = qm[int(s)].copy()
            if int(s) in nudges:
                q = (q + nudges[int(s)]).astype(np.float32)
            fvd[i, 0::3] = q
            fvd[i, 1::3] = 0.5 + i
            fvd[i, 2::3] = (depth or {}).get(int(s), 1.0 + 0.01 * int(s))
        gids = np.asarray([s for s in ids if int(s) in nudges], dtype=np.int64)
        nud = np.zeros((gids.size, NTS + 1), dtype=np.float32)
        nud[:, 0] = 1e6  # column 0 (initial condition) must be ignored
        for row, gid in enumerate(gids.tolist()):
            nud[row, 1:] = nudges[gid]
        out.append((ids, fvd, 0, (gids, np.zeros(gids.size), np.zeros(gids.size)), (), (), None, (), nud, ()))
    return out


def as_frame(results, col=0):
    parts = [pd.DataFrame(r[1][:, col::3], index=r[0]) for r in results]
    return pd.concat(parts).sort_index()


def reference_increments(rt, qm, dq_by_site):
    """Increments the spread should apply, from the core.

    Every tree's gage counts with its pre-nudge (modeled) flow, also where it is a
    pruned neighbour of another tree (the NGWPC background), so the reference uses
    the model flows everywhere.
    """
    segs = rt.needed_segs
    row = {int(s): i for i, s in enumerate(segs)}
    packed, _ = pack_trees(rt.trees, row)
    q = np.vstack([qm[int(s)] for s in segs])
    dq = np.vstack([dq_by_site.get(s, np.zeros(NTS)) for s in packed.sites])
    inc = spread_increments(packed, q, dq, split_floor_cms=rt.options.split_floor_cms)
    return {int(segs[r]): inc[k] for k, r in enumerate(packed.rows)}


def test_apply_matches_core_and_touches_only_tree_members():
    rng = np.random.default_rng(7)
    qm = model_flows(rng)
    nudges = {1: rng.uniform(-5, 5, NTS).astype(np.float32), 4: rng.uniform(-3, 8, NTS).astype(np.float32)}
    rt = build()
    res = make_results(qm, nudges)
    before = as_frame(res)
    vel_before, dep_before = as_frame(res, 1), as_frame(res, 2)
    assert rt.apply(res, dt=300.0, final=False) is False
    after = as_frame(res)
    ref = reference_increments(rt, qm, {"A": nudges[1], "B": nudges[4]})
    for s in ALL_SEGS:
        delta = after.loc[s].to_numpy(np.float64) - before.loc[s].to_numpy(np.float64)
        if s in ref and s not in GAGE_SEG.values():
            expect = np.maximum(qm[s].astype(np.float64) + ref[s], 0.0) - qm[s]
            np.testing.assert_allclose(delta, expect, rtol=1e-5, atol=1e-4)
        else:  # gages (nudging owns them), lake node, C's tree (no correction)
            assert np.array_equal(after.loc[s].to_numpy(), before.loc[s].to_numpy()), s
    assert np.array_equal(as_frame(res, 1).to_numpy(), vel_before.to_numpy())  # velocity untouched
    for s in (900, 11) + tuple(GAGE_SEG.values()):  # lake and gages: depth / elevation untouched
        assert np.array_equal(as_frame(res, 2).loc[s].to_numpy(), dep_before.loc[s].to_numpy())
    st = rt.stats[-1]
    assert st.n_trees_with_increment == 2 and st.n_segments_changed > 0 and not st.final
    applied = sum((after.loc[s].to_numpy(np.float64) - before.loc[s].to_numpy(np.float64)).sum() for s in ALL_SEGS)
    assert math.isclose(st.injected_volume_m3, applied * 300.0, rel_tol=1e-9, abs_tol=1e-6)


def test_linear_chain_closed_form():
    rng = np.random.default_rng(3)
    qm = model_flows(rng)
    dq = np.linspace(-2.0, 4.0, NTS).astype(np.float32)
    rt = build(theta={"default": 0.7})
    res = make_results(qm, {6: dq})
    before = as_frame(res)
    rt.apply(res, dt=300.0, final=False)
    after = as_frame(res)
    a = total_area()
    for s in (7, 8):
        expect = np.maximum(qm[s] + dq.astype(np.float64) * (a[s] / a[6]) ** 0.7, 0.0) - qm[s]
        got = after.loc[s].to_numpy(np.float64) - before.loc[s].to_numpy(np.float64)
        np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-4)


def test_confluence_split_uses_pre_nudge_gage_flow():
    """Huge nudge at A: the split must read A's modeled flow, not the nudged one."""
    rng = np.random.default_rng(11)
    qm = model_flows(rng)
    dq = np.full(NTS, 1000.0, dtype=np.float32)
    rt = build(observed=("A",))
    res = make_results(qm, {1: dq})
    before = as_frame(res)
    rt.apply(res, dt=60.0, final=False)
    after = as_frame(res)
    ref = reference_increments(rt, qm, {"A": dq})
    got = after.loc[9].to_numpy(np.float64) - before.loc[9].to_numpy(np.float64)
    np.testing.assert_allclose(got, np.maximum(qm[9] + ref[9], 0.0) - qm[9], rtol=1e-5, atol=1e-3)


def test_pruned_gage_neighbour_counts_with_its_modeled_flow():
    """B (4) is a pruned neighbour of A's tree at confluence 3; a huge nudge at B
    must not change the share of branch 9 (the split reads B's modeled flow)."""
    rng = np.random.default_rng(13)
    qm = model_flows(rng)
    dq_a = np.full(NTS, 10.0, dtype=np.float32)
    deltas = []
    for dq_b in (np.zeros(NTS, dtype=np.float32), np.full(NTS, 500.0, dtype=np.float32)):
        rt = build()
        res = make_results(qm, {1: dq_a, 4: dq_b})
        before = as_frame(res)
        rt.apply(res, dt=300.0, final=False)
        deltas.append(as_frame(res).loc[9].to_numpy(np.float64) - before.loc[9].to_numpy(np.float64))
    np.testing.assert_allclose(deltas[1], deltas[0], rtol=1e-6, atol=1e-5)
    assert np.abs(deltas[0]).max() > 0


def test_nudge_column_zero_is_ignored():
    rng = np.random.default_rng(5)
    qm = model_flows(rng)
    rt = build()
    res = make_results(qm, {4: np.zeros(NTS, dtype=np.float32)})  # column 0 is 1e6, the rest 0
    before = as_frame(res)
    rt.apply(res, dt=300.0, final=True)
    assert np.array_equal(as_frame(res).to_numpy(), before.to_numpy())
    assert rt.stats[-1].n_segments_changed == 0


def test_depth_follows_flow():
    rng = np.random.default_rng(9)
    qm = model_flows(rng)
    nud = np.asarray([0.0, 5.0, -5.0, 40.0, -40.0, 0.0], dtype=np.float32)
    rt = build(observed=("C",))
    res = make_results(qm, {6: nud}, depth={7: 2.0, 8: 3.0})
    q0, d0 = as_frame(res), as_frame(res, 2)
    rt.apply(res, dt=300.0, final=False)
    q1, d1 = as_frame(res), as_frame(res, 2)
    for s in (7, 8):
        qo, qn = q0.loc[s].to_numpy(np.float64), q1.loc[s].to_numpy(np.float64)
        ratio = np.clip(np.divide(qn, qo, out=np.ones_like(qo), where=qo > 1e-6), 1 / DEPTH_RATIO_CAP, DEPTH_RATIO_CAP)
        live = (qo > 1e-6) & (qn > 1e-6)
        expect = np.where(live, d0.loc[s].to_numpy(np.float64) * ratio ** DEPTH_EXPONENT, d0.loc[s].to_numpy())
        np.testing.assert_allclose(d1.loc[s].to_numpy(np.float64), expect, rtol=1e-6)
        assert d1.loc[s].iloc[0] == d0.loc[s].iloc[0]  # no correction at t=0 -> depth unchanged
    assert rt.stats[-1].n_depth_updates > 0


def test_final_flag_return_value():
    rng = np.random.default_rng(1)
    qm = model_flows(rng)
    rt = build()
    assert rt.apply(make_results(qm, {1: np.ones(NTS, dtype=np.float32)}), dt=300.0, final=True) is True
    assert rt.apply(make_results(qm, {}), dt=300.0, final=True) is False
    assert [s.loop for s in rt.stats] == [0, 1] and rt.stats[0].final


def test_apply_with_a_tree_segment_missing_raises():
    rng = np.random.default_rng(2)
    qm = model_flows(rng)
    rt = build()
    res = make_results(qm, {}, split=((1, 2, 3, 9, 10, 11, 900), (4, 5, 6, 7)))  # 8 missing
    pytest.raises(RuntimeError, rt.apply, res, dt=300.0, final=False).match("no row")


def test_apply_with_duplicate_segment_raises():
    rng = np.random.default_rng(2)
    qm = model_flows(rng)
    rt = build()
    res = make_results(qm, {}, split=((1, 2, 3, 9, 10, 11, 900, 5), (4, 5, 6, 7, 8)))
    with pytest.raises(RuntimeError, match="more than one result"):
        rt.apply(res, dt=300.0, final=False)


def test_apply_with_bad_nudge_shape_raises():
    rng = np.random.default_rng(2)
    qm = model_flows(rng)
    rt = build()
    res = make_results(qm, {4: np.ones(NTS, dtype=np.float32)})
    r = list(res[1])
    r[8] = r[8][:, :-1]
    res[1] = tuple(r)
    with pytest.raises(RuntimeError, match="nudge"):
        rt.apply(res, dt=300.0, final=False)


def test_chunking_does_not_change_results():
    rng = np.random.default_rng(4)
    qm = model_flows(rng)
    nudges = {1: rng.uniform(-5, 5, NTS).astype(np.float32), 6: rng.uniform(-5, 5, NTS).astype(np.float32)}
    outs = []
    for chunk in (None, 0, 1, 4):
        params = {} if chunk is None else {"spread_chunk_timesteps": chunk}
        rt = build(**params)
        res = make_results(qm, nudges)
        rt.apply(res, dt=300.0, final=False)
        outs.append((as_frame(res).to_numpy(), as_frame(res, 2).to_numpy()))
    for q, d in outs[1:]:
        assert np.array_equal(q, outs[0][0]) and np.array_equal(d, outs[0][1])
