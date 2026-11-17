#!/usr/bin/env python3
"""Area-scaled streamflow DA: gage-tree diagnostic on an HFv4.0 domain.

Builds the HYFeaturesNetwork from each t-route YAML (no routing, no forcing
read), then the scaling-DA gage trees, and reports per mode: trees, nodes,
drops by reason, area inversions, drainage-area flags, build time and memory.
Each YAML runs in its own subprocess so peak RSS is per mode.

Usage:

    python scaling_da_hfv4_diagnostic.py conus_flowpaths.yaml conus_flowlines.yaml \
        --geo-file /path/vpus/01/vpu01.gpkg --out vpu01_scaling_da_diag.json

`--geo-file` overrides `geo_file_path` so the YAMLs need no edits. Config
is validated non-strictly (missing forcing/TimeSlice folders are fine).
Writes `<out stem>_<mode>_gages.csv` (the per-gage report) next to `--out`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path


def rss_mb() -> dict:
    out = {}
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    k, v = line.split(":")
                    out[k] = round(int(v.split()[0]) / 1024, 1)
    except OSError:
        pass
    return out


def run_single(yaml_path: str, geo_file: str, max_reach_km: float, out_csv: str) -> dict:
    import numpy as np
    import yaml

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("troute.scaling_da").setLevel(logging.INFO)

    from troute.config import Config
    from troute.HYFeaturesNetwork import HYFeaturesNetwork
    from troute.scaling_da import pack_trees
    from troute.scaling_da.hyfeatures import build_hyfeatures_trees, results_seg_to_row, routed_segment_ids

    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)
    if geo_file:
        data["network_topology_parameters"]["supernetwork_parameters"]["geo_file_path"] = geo_file
    cfg = Config(**data).dict()
    cp, nt = cfg["compute_parameters"], cfg["network_topology_parameters"]
    rep = {"yaml": str(Path(yaml_path).resolve()),
           "geo_file": str(nt["supernetwork_parameters"]["geo_file_path"]),
           "rss_start_mb": rss_mb()}

    t0 = time.time()
    net = HYFeaturesNetwork(
        nt["supernetwork_parameters"], nt["waterbody_parameters"], cp["data_assimilation_parameters"],
        cp["restart_parameters"], cp, cp["forcing_parameters"], cp["hybrid_parameters"],
        nt["preprocessing_parameters"], cfg["output_parameters"], verbose=False, showtiming=False,
    )
    rep["network_build_s"] = round(time.time() - t0, 2)
    rep["rss_after_network_mb"] = rss_mb()
    rep["side_tables_mb"] = round(
        (net.segment_attributes.memory_usage(deep=True).sum()
         + net.gage_hydrolocations.memory_usage(deep=True).sum()) / 2**20, 2)

    t1 = time.time()
    b = build_hyfeatures_trees(net, max_reach_km=max_reach_km)
    rep["tree_build_s"] = round(time.time() - t1, 2)
    routed = routed_segment_ids(net)
    t2 = time.time()
    packed, pack_dropped = pack_trees(b.result.trees, results_seg_to_row(routed))
    rep["pack_s"] = round(time.time() - t2, 2)
    rep["rss_end_mb"] = rss_mb()

    m = b.mappings
    sizes = np.array([t.n_nodes for t in b.result.trees.values()] or [0])
    by_reason = {}
    for r in b.dropped.values():
        by_reason[r] = by_reason.get(r, 0) + 1
    # #887 crosswalk: nwis sites replaced by another id on the same segment (DA-01 issue)
    hl = net.gage_hydrolocations
    xw_sites = {str(s) for s in (net.gages or {}).get("gages", {}).values()}
    nwis = hl[hl["type"] == "nwis"]
    other_on_fp = set(hl.loc[(hl["type"] != "nwis") & hl["site"].astype(str).isin(xw_sites), "flowpath_id"])
    shadowed = sorted(
        s for s, fp in zip(nwis["site"].astype(str), nwis["flowpath_id"])
        if s not in xw_sites and fp in other_on_fp
    )
    rep.update({
        "mode": m.mode,
        "routed_nodes": int(routed.size),
        "lake_segments": len(m.lake_segments),
        "gage_sites": int(len(b.gage_report)),
        "excluded_types": m.excluded_types,
        "mapped": len(m.gage_segments),
        "trees": len(b.result.trees),
        "tree_nodes": int(sizes.sum()),
        "tree_size_max": int(sizes.max()),
        "tree_size_median": float(np.median(sizes)),
        "routed_nodes_in_trees_pct": round(100.0 * sizes.sum() / max(routed.size, 1), 1),
        "drops_by_reason": dict(sorted(by_reason.items())),
        "n_capped_nodes": int(sum(b.result.n_capped.values())),
        "n_area_inversions_network": m.n_area_inversions_network,
        "n_area_inversions_in_trees": int(packed.n_area_inversions),
        "pack_dropped": pack_dropped,
        "gages_flagged_poi_da": int(b.gage_report["flagged"].sum()),
        "nwis_shadowed_in_887_crosswalk": {"count": len(shadowed), "sites": shadowed[:25]},
        "max_reach_km": max_reach_km,
    })
    b.gage_report.to_csv(out_csv, index=False)
    rep["gage_report_csv"] = out_csv
    return rep


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("yaml", nargs="+", help="t-route V4 YAML(s), e.g. one flowpath and one flowline config")
    p.add_argument("--geo-file", default="", help="override geo_file_path")
    p.add_argument("--max-reach-km", type=float, default=200.0)
    p.add_argument("--out", default="scaling_da_diag.json")
    p.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--csv", default="", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.single:
        print(json.dumps(run_single(a.yaml[0], a.geo_file, a.max_reach_km, a.csv), default=str))
        return 0
    out = Path(a.out)
    reports = []
    for y in a.yaml:
        tag = Path(y).stem
        csv = str(out.with_name(f"{out.stem}_{tag}_gages.csv"))
        cmd = [sys.executable, os.path.abspath(__file__), y, "--single", "--csv", csv,
               "--max-reach-km", str(a.max_reach_km)] + (["--geo-file", a.geo_file] if a.geo_file else [])
        print(f"== {y}", file=sys.stderr, flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            reports.append({"yaml": y, "error": f"exit {proc.returncode}"})
            continue
        r = json.loads(proc.stdout.strip().splitlines()[-1])
        r["wall_s"] = round(time.time() - t0, 1)
        reports.append(r)
    out.write_text(json.dumps(reports, indent=2, default=str) + "\n")
    keys = ["mode", "routed_nodes", "gage_sites", "excluded_types", "mapped", "trees", "tree_nodes",
            "tree_size_max", "routed_nodes_in_trees_pct", "drops_by_reason", "n_capped_nodes",
            "n_area_inversions_network", "n_area_inversions_in_trees", "gages_flagged_poi_da",
            "nwis_shadowed_in_887_crosswalk", "network_build_s", "tree_build_s", "pack_s",
            "rss_after_network_mb", "rss_end_mb", "side_tables_mb", "wall_s"]
    for r in reports:
        print(f"\n== {r.get('yaml')}")
        for k in keys:
            if k in r:
                print(f"  {k}: {r[k]}")
        if "error" in r:
            print(f"  error: {r['error']}")
    print(f"\nwrote {out}")
    return 0 if all("error" not in r for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
