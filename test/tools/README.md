# HFv4.0 test tools

Scripts that generate hydrofabric-derived test data and diagnostics. The data
they produce is **not committed**: tests that need it skip
cleanly when it is absent. Candidates for `t-route-utils`.

| Script | Purpose |
|---|---|
| `hfv4_subset.py` | Subset an HFv4.0 GeoPackage: `vpu` (by `vpuid`), `candidates` (rank fixture outlets), `upstream` (basin above outlet flowpaths). Stdlib only; rows copied verbatim; sidecar `<out>.json` records source, arguments and per-layer content hashes. |
| `scaling_da_hfv4_diagnostic.py` | Area-scaled streamflow DA gage trees on a domain (both modes): trees, nodes, drops by reason, area inversions, drainage-area flags, build time and memory. |
| `scaling_da_smoke.py` | Area-scaled streamflow DA smoke run: open loop vs nudging vs scaling through `main_v04` from one base YAML, one subprocess per arm; wall time, scaling setup and spread time, peak RSS, segments changed, changes outside trees (must be 0). |

## Scaling-DA fixture

```bash
# 1. VPU subset of the CONUS file (optional; `candidates`/`upstream` also accept the CONUS file)
python test/tools/hfv4_subset.py vpu <superconus.gpkg> --vpu 01 -o vpu01.gpkg

# 2. pick an outlet: >= 3 nwis gages (some nested), a lake, a confluence, non-routeable flowlines
python test/tools/hfv4_subset.py candidates vpu01.gpkg --csv candidates.csv

# 3. cut the fixture (record the outlet and source; the sidecar JSON does this too)
python test/tools/hfv4_subset.py upstream vpu01.gpkg --outlet <fp-id> -o hfv4_fixture.gpkg

# 4. run the fixture tests
TROUTE_HFV4_FIXTURE=$PWD/hfv4_fixture.gpkg pytest src/troute-network/tests/scaling_da -m hfv4_fixture -v
```

The fixture tests include `test_scaling_da_integration.py`, which runs `main_v04`
on the fixture in both modes with synthetic inputs written to a temporary
directory (NextGen-style forcing NetCDF, USGS TimeSlices of 1.3 x the open-loop
flow at each gage). It needs `netCDF4` and `xarray` and takes a few minutes.

## Scaling-DA smoke run

```bash
# base YAML: -V4 config with forcing, usgs_timeslices_folder and (optionally)
# streamflow_scaling_parameters; the arms only change the streamflow DA flags
python test/tools/scaling_da_smoke.py run --config vpu01_flowpath.yaml --out smoke_fp
python test/tools/scaling_da_smoke.py compare --out smoke_fp   # re-run the comparison only
```

`summary.md`/`summary.json` in `--out`; per-arm `meta.json`, saved flows, and for
the scaling arm `gages.csv` (per-gage status) and `loops.csv` (per-loop statistics).

Regenerating from the same source and outlet gives the same rows: compare
`output.content_sha256_combined` in the sidecar JSON.

Record the fixture in use (source file, outlets and `content_sha256_combined`)
with the change that depends on it, e.g. in its PR description.

## Notes

- t-route's `read_geopkg` picks layers by regex in listing order, and its
  flowline pattern also matches `flowline-attributes`. `hfv4_subset.py` writes
  each `*-attributes` layer after its base layer and checks the result with
  `fiona.listlayers`; it exits with status 1 if t-route would pick the wrong layer.
- A `vpu` subset keeps cross-VPU links as dangling ids; the post-write checks
  report them (`flowpath_toid_not_in_nexus`, `nexus_toid_outside_subset`).
