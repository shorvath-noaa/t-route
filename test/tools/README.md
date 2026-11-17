# HFv4.0 test tools

Scripts that generate hydrofabric-derived test data and diagnostics. The data
they produce is **not committed**: tests that need it skip
cleanly when it is absent. Candidates for `t-route-utils`.

| Script | Purpose |
|---|---|
| `hfv4_subset.py` | Subset an HFv4.0 GeoPackage: `vpu` (by `vpuid`), `candidates` (rank fixture outlets), `upstream` (basin above outlet flowpaths). Stdlib only; rows copied verbatim; sidecar `<out>.json` records source, arguments and per-layer content hashes. |
| `scaling_da_hfv4_diagnostic.py` | Area-scaled streamflow DA gage trees on a domain (both modes): trees, nodes, drops by reason, area inversions, drainage-area flags, build time and memory. |

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
