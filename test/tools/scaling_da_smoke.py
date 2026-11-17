#!/usr/bin/env python3
"""Area-scaled streamflow DA smoke run: open loop vs nudging vs scaling on one domain.

Runs `main_v04` once per arm, each in its own subprocess (so peak RSS is per arm),
from one base YAML that differs between arms only in the streamflow DA flags:

* `open_loop`: `streamflow_nudging: False`, `streamflow_scaling: False`;
* `nudging`:   `streamflow_nudging: True`,  `streamflow_scaling: False`;
* `scaling`:   `streamflow_nudging: True`,  `streamflow_scaling: True`
  (`streamflow_scaling_parameters` from the base YAML, plus `--holdout`).

Each arm saves the published streamflow every `--save-every` routing steps, its
timing (total, scaling setup, upstream spread) and peak RSS, and for the scaling arm
the per-gage report, per-loop statistics and the tree member ids. `compare` then
writes `summary.json` and `summary.md`.

Usage:

    python test/tools/scaling_da_smoke.py run   --config vpu01_flowpath.yaml --out smoke_fp
    python test/tools/scaling_da_smoke.py compare --out smoke_fp

`--geo-file` overrides `geo_file_path`. Output writers in the base YAML
(`netcdf_output`, `stream_output`, `lite_restart`, ...) are removed so that
arms time routing and DA only; pass `--keep-output` to keep them. Peak RSS is the
driver process's (`VmHWM`); joblib worker processes (`cpu_pool > 1`) are not
included.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ARMS = ("open_loop", "nudging", "scaling")
OUTPUT_KEYS = ("netcdf_output", "stream_output", "lite_restart", "chrtout_output", "chanobs_output",
               "csv_output", "parquet_output", "lakeout_output", "test_output", "wrf_hydro_parity_check",
               "lastobs_output")


def vm_hwm_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def arm_config(base: dict, arm: str, geo_file: str, holdout: str, keep_output: bool) -> dict:
    cfg = copy.deepcopy(base)
    if geo_file:
        cfg["network_topology_parameters"]["supernetwork_parameters"]["geo_file_path"] = geo_file
    if not keep_output:
        out = cfg.get("output_parameters") or {}
        for k in OUTPUT_KEYS:
            out.pop(k, None)
        cfg["output_parameters"] = out
    da = cfg["compute_parameters"].setdefault("data_assimilation_parameters", {})
    stream_da = dict(da.get("streamflow_da") or {})
    stream_da.update(streamflow_nudging=arm != "open_loop", streamflow_scaling=arm == "scaling")
    if arm == "scaling" and holdout:
        stream_da.setdefault("streamflow_scaling_parameters", {})["holdout_sites_file"] = holdout
    if arm != "scaling":
        stream_da.pop("streamflow_scaling_parameters", None)
    da["streamflow_da"] = stream_da
    return cfg


def child(args) -> int:
    import numpy as np
    import pandas as pd
    import yaml

    out = Path(args.out) / args.arm
    os.makedirs(out, exist_ok=True)
    base = yaml.safe_load(Path(args.config).read_text())
    cfg = arm_config(base, args.arm, args.geo_file, args.holdout, args.keep_output)
    yfile = out / "config.yaml"
    yfile.write_text(yaml.safe_dump(cfg))

    import nwm_routing.__main__ as main
    from troute.scaling_da.runtime import ScalingDARuntime

    logging.getLogger("troute.scaling_da").setLevel(logging.INFO)
    every = max(1, int(args.save_every))
    chunks, ids_seen = [], None
    timers = {"setup_s": 0.0, "spread_s": 0.0}
    holder = {}
    orig_build = ScalingDARuntime.build.__func__
    orig_apply = ScalingDARuntime.apply

    def collect(run, run_results, *a, **k):
        nonlocal ids_seen
        ids = np.concatenate([np.asarray(res[0], dtype=np.int64) for res in run_results])
        q = np.vstack([res[1][:, 0::3] for res in run_results])
        order = np.argsort(ids)
        ids, q = ids[order], q[order]
        if ids_seen is None:
            ids_seen = ids
        elif not np.array_equal(ids, ids_seen):
            raise RuntimeError("segment set changed between loops")
        chunks.append(q[:, every - 1::every].astype(np.float32))

    def build(cls, *a, **k):
        t = time.perf_counter()
        holder["rt"] = orig_build(cls, *a, **k)
        timers["setup_s"] += time.perf_counter() - t
        return holder["rt"]

    def apply(self, *a, **k):
        t = time.perf_counter()
        r = orig_apply(self, *a, **k)
        timers["spread_s"] += time.perf_counter() - t
        return r

    main.nwm_output_generator = collect
    ScalingDARuntime.build = classmethod(build)
    ScalingDARuntime.apply = apply
    t0 = time.perf_counter()
    main.main_v04(["-f", str(yfile)])
    wall = time.perf_counter() - t0

    flow = np.concatenate(chunks, axis=1) if chunks else np.empty((0, 0), np.float32)
    np.save(out / "flow.npy", flow)
    np.save(out / "ids.npy", ids_seen if ids_seen is not None else np.empty(0, np.int64))
    meta = {"arm": args.arm, "wall_s": round(wall, 2), "peak_rss_mb": vm_hwm_mb(),
            "save_every": every, "n_segments": int(flow.shape[0]), "n_saved_steps": int(flow.shape[1]),
            "setup_s": round(timers["setup_s"], 3), "spread_s": round(timers["spread_s"], 3)}
    rt = holder.get("rt")
    if rt is not None:
        rt.report.to_csv(out / "gages.csv", index=False)
        pd.DataFrame([s.__dict__ for s in rt.stats]).to_csv(out / "loops.csv", index=False)
        members = sorted({int(s) for t in rt.trees.values() for s in t.seg_ids})
        gages = sorted({int(t.gage_segment) for t in rt.trees.values()})
        np.save(out / "tree_members.npy", np.asarray(members, dtype=np.int64))
        np.save(out / "tree_gages.npy", np.asarray(gages, dtype=np.int64))
        meta.update(n_sources=len(rt.sources), n_trees=len(rt.trees), n_tree_segments=len(members),
                    n_held_out=len(rt.held_out), n_no_observations=len(rt.no_observations),
                    injected_volume_m3=float(sum(s.injected_volume_m3 for s in rt.stats)))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    return 0


def run(args) -> int:
    for arm in args.arms:
        cmd = [sys.executable, __file__, "_child", "--arm", arm, "--config", args.config, "--out", args.out,
               "--save-every", str(args.save_every)]
        if args.geo_file:
            cmd += ["--geo-file", args.geo_file]
        if args.holdout:
            cmd += ["--holdout", args.holdout]
        if args.keep_output:
            cmd += ["--keep-output"]
        print(f"== {arm}: {' '.join(cmd)}", flush=True)
        log = Path(args.out) / f"{arm}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
        print(f"   exit {rc}; log {log}", flush=True)
        if rc:
            return rc
    return compare(args)


def _load(out: Path, arm: str):
    import numpy as np

    d = out / arm
    if not (d / "meta.json").is_file():
        return None
    return {
        "meta": json.loads((d / "meta.json").read_text()),
        "ids": np.load(d / "ids.npy"),
        "flow": np.load(d / "flow.npy", mmap_mode="r"),
        "members": np.load(d / "tree_members.npy") if (d / "tree_members.npy").is_file() else None,
        "gages": np.load(d / "tree_gages.npy") if (d / "tree_gages.npy").is_file() else None,
    }


def _diff(a, b, tol):
    import numpy as np

    if not np.array_equal(a["ids"], b["ids"]) or a["flow"].shape != b["flow"].shape:
        raise RuntimeError(f"{a['meta']['arm']} and {b['meta']['arm']} saved different segments or steps")
    d = np.abs(np.asarray(a["flow"], dtype=np.float64) - np.asarray(b["flow"], dtype=np.float64))
    changed = (d > tol).any(axis=1)
    rel = d / np.maximum(np.abs(np.asarray(b["flow"], dtype=np.float64)), 1e-6)
    return d, changed, rel


def compare(args) -> int:
    import numpy as np

    out = Path(args.out)
    arms = {a: _load(out, a) for a in ARMS}
    summary = {"arms": {a: v["meta"] for a, v in arms.items() if v}, "tolerance_cms": args.tol}
    nd, sc, ol = arms["nudging"], arms["scaling"], arms["open_loop"]
    if nd and ol:
        _, ch, _ = _diff(nd, ol, args.tol)
        summary["nudging_vs_open_loop"] = {"segments_changed": int(ch.sum()), "of": int(ch.size)}
    if nd and sc:
        d, ch, rel = _diff(sc, nd, args.tol)
        ids = sc["ids"]
        members = set(sc["members"].tolist()) if sc["members"] is not None else set()
        gages = set(sc["gages"].tolist()) if sc["gages"] is not None else set()
        in_tree = np.isin(ids, sorted(members - gages))
        dmax = d.max(axis=1)
        summary["scaling_vs_nudging"] = {
            "segments_changed": int(ch.sum()),
            "of": int(ch.size),
            "pct_changed": round(100.0 * ch.mean(), 2) if ch.size else 0.0,
            "tree_segments_excluding_gages": int(in_tree.sum()),
            "changed_outside_trees": int((ch & ~in_tree).sum()),
            "max_abs_change_cms": float(dmax.max()) if dmax.size else 0.0,
            "median_max_abs_change_cms_changed": float(np.median(dmax[ch])) if ch.any() else 0.0,
            "median_max_rel_change_changed": float(np.median(rel.max(axis=1)[ch])) if ch.any() else 0.0,
            "gage_segments_changed": int((ch & np.isin(ids, sorted(gages))).sum()),
        }
        w_nd, w_sc = nd["meta"]["wall_s"], sc["meta"]["wall_s"]
        summary["scaling_vs_nudging"]["wall_overhead_pct"] = round(100.0 * (w_sc - w_nd) / w_nd, 1) if w_nd else None
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["| arm | wall s | peak RSS MB | scaling setup s | spread s |", "|---|---|---|---|---|"]
    for name, arm in arms.items():
        if not arm:
            continue
        m = arm["meta"]
        lines.append(f"| {name} | {m['wall_s']} | {m['peak_rss_mb']} | {m.get('setup_s', 0)} | {m.get('spread_s', 0)} |")
    for k in ("nudging_vs_open_loop", "scaling_vs_nudging"):
        if k in summary:
            lines += ["", f"**{k}**", ""] + [f"- {kk}: {vv}" for kk, vv in summary[k].items()]
    if sc:
        lines += ["", "**scaling setup**", ""] + [
            f"- {k}: {sc['meta'][k]}" for k in ("n_sources", "n_trees", "n_tree_segments", "n_held_out",
                                                 "n_no_observations", "injected_volume_m3") if k in sc["meta"]]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sv = summary.get("scaling_vs_nudging", {})
    if sv.get("changed_outside_trees") or sv.get("gage_segments_changed"):
        print("WARNING: scaling changed segments outside its trees or at gages", file=sys.stderr)
        return 2
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "_child"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True, help="base t-route YAML (-V4)")
        s.add_argument("--out", required=True, help="output directory")
        s.add_argument("--geo-file", default="", help="override geo_file_path")
        s.add_argument("--holdout", default="", help="holdout_sites_file for the scaling arm")
        s.add_argument("--save-every", type=int, default=12, help="save every Nth routing step (default 12)")
        s.add_argument("--keep-output", action="store_true", help="keep the YAML's output writers")
        s.add_argument("--tol", type=float, default=1e-4, help="change threshold (m3/s) for compare")
        if name == "run":
            s.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
        else:
            s.add_argument("--arm", required=True, choices=ARMS)
    s = sub.add_parser("compare")
    s.add_argument("--out", required=True)
    s.add_argument("--tol", type=float, default=1e-4)
    args = p.parse_args(argv)
    return {"run": run, "_child": child, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
