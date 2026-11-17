#!/usr/bin/env python3
"""Subset a Lynker-Spatial HFv4.0 GeoPackage (e.g. `superconus_ngen_edited.gpkg`).

Subcommands
-----------
vpu         Copy every layer, keeping rows whose `vpuid` is in `--vpu`.
candidates  Rank outlet flowpaths whose upstream basin suits a scaling-DA test
            fixture (gages in series and on branches, a lake, a confluence,
            non-routeable flowlines, a `tnx-` terminal if possible).
upstream    Copy the basin upstream of one or more outlet flowpaths (the fixture).

Design
------
* Standard library only (`sqlite3`): a GeoPackage is an SQLite database, so
  rows and geometry blobs are copied verbatim with their `fid`. No dtype
  round trip (`BOOLEAN`/`MEDIUMINT`/`TEXT` ids stay as declared), no
  GDAL needed to write, and the same inputs always give the same rows.
* The output is a valid GeoPackage: `gpkg_spatial_ref_sys`,
  `gpkg_contents`, `gpkg_geometry_columns`, `gpkg_extensions`,
  `gpkg_ogr_contents`, the R-tree spatial indexes (copied, ids = fids) and
  the source's attribute indexes. If this Python's SQLite lacks the R-tree
  module the spatial indexes are skipped (still a valid GeoPackage).
* Layers are written in the source's `gpkg_contents` order, except that a
  `*-attributes` layer always follows its base layer. t-route's
  `read_geopkg` matches layers by regex in listing order, and its
  `flowlines` pattern also matches `flowline-attributes`.
* A sidecar `<output>.json` records the tool version, arguments, source file
  (path, size, mtime, optional sha256) and per-layer row counts and content
  hashes, so a fixture can be regenerated and checked without committing it.
* The output is written to `<output>.partial` and renamed when complete.

Examples
--------
    # one VPU (or several into one file: --vpu 01 --vpu 02)
    python hfv4_subset.py vpu superconus_ngen_edited.gpkg --vpu 01 -o 01/vpu01.gpkg

    # every VPU, one file each (what the old per-VPU loop did)
    for v in 01 02 03N 03S 03W 04 05 06 07 08 09 10L 10U 11 12 13 14 15 16 17 18; do
        python hfv4_subset.py vpu superconus_ngen_edited.gpkg --vpu $v -o $v/vpu$v.gpkg
    done

    # fixture search and cut
    python hfv4_subset.py candidates 01/vpu01.gpkg --csv candidates.csv
    python hfv4_subset.py upstream 01/vpu01.gpkg --outlet fp-12345 -o hfv4_fixture.gpkg

Field names follow `superconus_ngen_edited.gpkg` (HFv4.0); they may change
in HFv4.1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sqlite3
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCRIPT_VERSION = "1.0.0"
TOOL_NAME = "hfv4_subset.py"

# t-route read_geopkg layer patterns (HYFeaturesNetwork.py at HFv3 e5f7dab).
TROUTE_LAYER_PATTERNS = {
    "flowpath": {
        "flowpaths": r"flow[-_]?paths?",
        "flowpath_attributes": r"flow[-_]?path[-_]?attributes?",
    },
    "flowline": {
        "flowpaths": r"flow[-_]?lines?|flow[-_]?lines?",
        "flowpath_attributes": r"flow[-_]?line[-_]?attributes?",
    },
}
TROUTE_EXPECTED = {
    "flowpath": {"flowpaths": "flowpaths", "flowpath_attributes": "flowpath-attributes"},
    "flowline": {"flowpaths": "flowlines", "flowpath_attributes": "flowline-attributes"},
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def qi(name: str) -> str:
    """Quote an SQL identifier (layer names contain '-')."""
    return '"' + name.replace('"', '""') + '"'


def open_ro(path: Path) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def columns(con: sqlite3.Connection, table: str, schema: str = "main") -> List[str]:
    return [r[1] for r in con.execute(f"PRAGMA {schema}.table_info({qi(table)})")]


def has_table(con: sqlite3.Connection, table: str, schema: str = "main") -> bool:
    return con.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type IN ('table','view') AND name=?", (table,)
    ).fetchone() is not None


def gpkg_layers(con: sqlite3.Connection, schema: str = "main") -> List[Tuple[str, str]]:
    """`(table_name, data_type)` of vector/attribute layers, in gpkg_contents rowid order."""
    return [
        (r[0], r[1])
        for r in con.execute(
            f"SELECT table_name, data_type FROM {schema}.gpkg_contents "
            "WHERE data_type IN ('features','attributes') ORDER BY rowid"
        )
    ]


def troute_safe_order(names: Sequence[str]) -> List[str]:
    """Keep order, but move `X-attributes` right after `X` (or `X`-singular+s)."""
    out = [n for n in names if not re.search(r"[-_]attributes?$", n, re.I)]
    for n in names:
        if n in out:
            continue
        base = re.sub(r"[-_]attributes?$", "", n, flags=re.I)
        cands = [i for i, m in enumerate(out) if m.lower() in (base.lower(), base.lower() + "s")]
        if cands:
            out.insert(cands[0] + 1, n)
        else:
            out.append(n)
    return out


def rtree_available() -> bool:
    try:
        c = sqlite3.connect(":memory:")
        c.execute("CREATE VIRTUAL TABLE t USING rtree(id, minx, maxx, miny, maxy)")
        c.close()
        return True
    except sqlite3.Error:
        return False


def file_sha256(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def table_digest(con: sqlite3.Connection, table: str) -> str:
    h = hashlib.sha256()
    for row in con.execute(f"SELECT * FROM {qi(table)} ORDER BY fid"):
        h.update(repr(row).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def split_refs(s: Optional[str]) -> List[str]:
    return [p.strip() for p in (s or "").split("|") if p.strip()]


# ---------------------------------------------------------------------------
# GeoPackage writer
# ---------------------------------------------------------------------------
class Selector:
    """WHERE clause (against `src."<layer>"`) plus parameters; None = skip layer."""

    def __init__(self, where: str, params: Sequence = ()):
        self.where = where
        self.params = tuple(params)


def write_subset(
    src_path: Path,
    out_path: Path,
    selectors: Dict[str, Selector],
    *,
    temp_ids: Optional[Dict[str, Iterable]] = None,
    overwrite: bool = False,
    spatial_index: bool = True,
) -> Dict[str, int]:
    """Copy the selected rows of each layer from `src_path` into a new GeoPackage.

    `temp_ids` creates `temp.<name>(id PRIMARY KEY)` tables that selectors
    may reference (e.g. `flowpath_id IN (SELECT id FROM temp.sel_fp)`).
    Returns `{layer: rows}`.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".partial")
    for p in (tmp, Path(str(tmp) + "-journal")):
        if p.exists():
            p.unlink()

    use_rtree = spatial_index and rtree_available()
    if spatial_index and not use_rtree:
        log("note: this SQLite has no R-tree module; spatial indexes are not copied")

    con = sqlite3.connect(Path(tmp).resolve().as_uri() + "?mode=rwc", uri=True)
    try:
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("ATTACH DATABASE ? AS src", (Path(src_path).resolve().as_uri() + "?mode=ro",))
        for pragma in ("application_id", "user_version"):
            v = con.execute(f"PRAGMA src.{pragma}").fetchone()[0]
            con.execute(f"PRAGMA main.{pragma}={int(v)}")

        def ddl(name: str, kind: str = "table") -> Optional[str]:
            r = con.execute(
                "SELECT sql FROM src.sqlite_master WHERE type=? AND name=?", (kind, name)
            ).fetchone()
            return r[0] if r else None

        for t in ("gpkg_spatial_ref_sys", "gpkg_contents", "gpkg_geometry_columns"):
            sql = ddl(t)
            if sql is None:
                raise SystemExit(f"{src_path} is not a GeoPackage (no {t})")
            con.execute(sql)
        con.execute("INSERT INTO main.gpkg_spatial_ref_sys SELECT * FROM src.gpkg_spatial_ref_sys")

        for name, ids in (temp_ids or {}).items():
            con.execute(f"CREATE TEMP TABLE {qi(name)} (id PRIMARY KEY) WITHOUT ROWID")
            con.executemany(
                f"INSERT OR IGNORE INTO temp.{qi(name)} VALUES (?)", ((i,) for i in ids)
            )

        src_order = [n for n, _ in gpkg_layers(con, "src")]
        order = troute_safe_order([n for n in src_order if n in selectors])
        counts: Dict[str, int] = {}
        rtree_tables: List[Tuple[str, str]] = []
        for layer in order:
            sel = selectors[layer]
            t0 = time.time()
            con.execute(ddl(layer))
            con.execute(
                f"INSERT INTO main.{qi(layer)} SELECT * FROM src.{qi(layer)} "
                f"WHERE {sel.where} ORDER BY fid",
                sel.params,
            )
            n = con.execute(f"SELECT count(*) FROM main.{qi(layer)}").fetchone()[0]
            counts[layer] = n
            con.execute(
                "INSERT INTO main.gpkg_contents SELECT * FROM src.gpkg_contents WHERE table_name=?",
                (layer,),
            )
            con.execute(
                "INSERT INTO main.gpkg_geometry_columns SELECT * FROM src.gpkg_geometry_columns "
                "WHERE table_name=?",
                (layer,),
            )
            for (sql,) in con.execute(
                "SELECT sql FROM src.sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL",
                (layer,),
            ).fetchall():
                con.execute(sql)
            geom = con.execute(
                "SELECT column_name FROM src.gpkg_geometry_columns WHERE table_name=?", (layer,)
            ).fetchone()
            if use_rtree and geom:
                rt = f"rtree_{layer}_{geom[0]}"
                rsql = ddl(rt)
                if rsql:
                    con.execute(rsql)
                    con.execute(
                        f"INSERT INTO main.{qi(rt)} SELECT r.* FROM src.{qi(rt)} r "
                        f"WHERE r.id IN (SELECT fid FROM main.{qi(layer)}) ORDER BY r.id"
                    )
                    rtree_tables.append((layer, rt))
            log(f"  {layer:<22} {n:>10,d} rows  ({time.time() - t0:.1f} s)")

        # Triggers (R-tree maintenance, GDAL feature counts). They reference
        # GDAL SQL functions, which only matter if someone writes to the file.
        copied = set(order)
        rtree_names = {rt for _, rt in rtree_tables}
        for name, tbl, sql in con.execute(
            "SELECT name, tbl_name, sql FROM src.sqlite_master WHERE type='trigger'"
        ).fetchall():
            if tbl not in copied:
                continue
            if "rtree_" in sql and not any(rt in sql for rt in rtree_names):
                continue  # maintains an R-tree we did not copy
            con.execute(sql)

        if has_table(con, "gpkg_extensions", "src"):
            con.execute(ddl("gpkg_extensions"))
            cols = columns(con, "gpkg_extensions", "src")
            for row in con.execute("SELECT * FROM src.gpkg_extensions").fetchall():
                rec = dict(zip(cols, row))
                tname, ext = rec.get("table_name"), rec.get("extension_name")
                if tname is not None and tname not in copied:
                    continue
                if ext == "gpkg_rtree_index" and tname not in {l for l, _ in rtree_tables}:
                    continue
                con.execute(
                    f"INSERT INTO main.gpkg_extensions VALUES ({','.join('?' * len(row))})", row
                )

        if has_table(con, "gpkg_ogr_contents", "src"):
            con.execute(ddl("gpkg_ogr_contents"))
            con.executemany(
                "INSERT INTO main.gpkg_ogr_contents (table_name, feature_count) VALUES (?, ?)",
                list(counts.items()),
            )

        for layer, rt in rtree_tables:
            con.execute(
                f"UPDATE main.gpkg_contents SET "
                f"min_x=(SELECT min(minx) FROM main.{qi(rt)}), max_x=(SELECT max(maxx) FROM main.{qi(rt)}), "
                f"min_y=(SELECT min(miny) FROM main.{qi(rt)}), max_y=(SELECT max(maxy) FROM main.{qi(rt)}) "
                f"WHERE table_name=?",
                (layer,),
            )
        con.commit()
        con.execute("DETACH DATABASE src")
        con.execute("VACUUM")
    finally:
        con.close()
    os.replace(tmp, out_path)
    return counts


# ---------------------------------------------------------------------------
# post-write checks and provenance
# ---------------------------------------------------------------------------
def layer_resolution(path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    """Which layer t-route's read_geopkg would pick, per mode.

    Uses `fiona.listlayers` (what t-route calls) when fiona imports,
    otherwise gpkg_contents rowid order.
    """
    try:
        import fiona  # type: ignore

        names = list(fiona.listlayers(str(path)))
        source = "fiona.listlayers"
    except Exception:
        con = open_ro(path)
        names = [n for n, _ in gpkg_layers(con)]
        con.close()
        source = "gpkg_contents order (fiona not importable)"
    res: Dict[str, Dict[str, Optional[str]]] = {"_listing": {"source": source, "order": ",".join(names)}}
    for mode, pats in TROUTE_LAYER_PATTERNS.items():
        res[mode] = {
            key: next((n for n in names if re.search(p, n, re.I)), None) for key, p in pats.items()
        }
    return res


def check_output(path: Path) -> Tuple[Dict[str, object], List[str]]:
    """Referential checks on the written file. Returns (summary, warnings)."""
    con = open_ro(path)
    warn: List[str] = []
    s: Dict[str, object] = {}

    def cols(t):
        return set(columns(con, t)) if has_table(con, t) else set()

    def one(sql, *p):
        return con.execute(sql, p).fetchone()[0]

    fp, nx, fl = cols("flowpaths"), cols("nexus"), cols("flowlines")
    if {"flowpath_toid"} <= fp and {"nexus_id"} <= nx:
        s["flowpath_toid_not_in_nexus"] = one(
            "SELECT count(*) FROM flowpaths WHERE flowpath_toid NOT IN (SELECT nexus_id FROM nexus)"
        )
        s["flowpaths_to_tnx"] = one("SELECT count(*) FROM flowpaths WHERE flowpath_toid LIKE 'tnx-%'")
    if {"nexus_toid"} <= nx and {"flowpath_id"} <= fp:
        s["nexus_toid_outside_subset"] = one(
            "SELECT count(*) FROM nexus WHERE nexus_toid IS NOT NULL "
            "AND nexus_toid NOT IN (SELECT flowpath_id FROM flowpaths)"
        )
    if {"flowline_toid", "flowline_id"} <= fl:
        s["flowline_toid_outside_subset"] = one(
            "SELECT count(*) FROM flowlines WHERE flowline_toid <> 0 "
            "AND flowline_toid NOT IN (SELECT flowline_id FROM flowlines)"
        )
    if "routeable" in fl:
        s["flowlines"] = one("SELECT count(*) FROM flowlines")
        s["flowlines_routeable"] = one("SELECT count(*) FROM flowlines WHERE routeable")
    for base, attr, key in (
        ("flowpaths", "flowpath-attributes", "flowpath_id"),
        ("flowlines", "flowline-attributes", "flowline_id"),
    ):
        if key in cols(base) and key in cols(attr):
            k = one(
                f"SELECT count(*) FROM {qi(base)} WHERE {key} NOT IN (SELECT {key} FROM {qi(attr)})"
            )
            s[f"{base}_without_attributes"] = k
            if k:
                warn.append(f"{k} {base} rows have no {attr} row (t-route inner-joins them away)")
    hl = cols("hydrolocations")
    if {"hl_class", "hl_reference"} <= hl:
        refs = [r[0] for r in con.execute("SELECT hl_reference FROM hydrolocations WHERE hl_class='gage'")]
        flat = [x for r in refs for x in split_refs(r)]
        s["gage_hydrolocations"] = len(refs)
        s["gage_refs_nwis"] = sum(x.startswith("nwis-") for x in flat)
        s["gage_refs_rfc"] = sum(x.startswith("rfc-") for x in flat)
    lk = cols("lakes")
    if "lake_id" in lk:
        s["lakes"] = one("SELECT count(*) FROM lakes")
        for t in ("flowpaths", "flowlines"):
            if "lake_id" in cols(t):
                k = one(
                    f"SELECT count(DISTINCT lake_id) FROM {qi(t)} WHERE lake_id IS NOT NULL "
                    "AND lake_id NOT IN (SELECT lake_id FROM lakes)"
                )
                s[f"{t}_lake_id_not_in_lakes"] = k
                if k:
                    warn.append(f"{k} lake_id values on {t} have no lakes row (t-route drops those waterbodies)")
    if "vpuid" in fp:
        s["vpuids"] = sorted(r[0] for r in con.execute("SELECT DISTINCT vpuid FROM flowpaths"))
    con.close()

    res = layer_resolution(path)
    s["troute_layer_resolution"] = res
    for mode, exp in TROUTE_EXPECTED.items():
        for key, want in exp.items():
            got = res[mode][key]
            if got is not None and got != want:
                warn.append(
                    f"t-route {mode} mode would read layer '{got}' as '{key}' (expected '{want}'); "
                    f"listing order: {res['_listing']['order']}"
                )
    try:  # optional GDAL cross-check of feature counts
        import pyogrio  # type: ignore

        s["pyogrio_feature_counts"] = {
            name: int(pyogrio.read_info(str(path), layer=name)["features"])
            for name, _ in pyogrio.list_layers(str(path))
        }
    except Exception as e:  # pragma: no cover - environment dependent
        s["pyogrio_feature_counts"] = f"skipped ({type(e).__name__})"
    return s, warn


def write_sidecar(
    out_path: Path, src_path: Path, args: argparse.Namespace, counts: Dict[str, int],
    extra: Dict[str, object], hash_source: bool,
) -> Path:
    st = Path(src_path).stat()
    con = open_ro(out_path)
    digests = {t: table_digest(con, t) for t in counts}
    con.close()
    combined = hashlib.sha256("".join(f"{t}:{d}\n" for t, d in sorted(digests.items())).encode()).hexdigest()
    meta = {
        "tool": TOOL_NAME,
        "tool_version": SCRIPT_VERSION,
        "command": args.cmd,
        "arguments": {k: v for k, v in vars(args).items() if k not in ("func", "cmd")},
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "path": str(Path(src_path).resolve()),
            "size_bytes": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256": file_sha256(src_path) if hash_source else None,
        },
        "output": {
            "path": str(Path(out_path).resolve()),
            "size_bytes": Path(out_path).stat().st_size,
            "rows": counts,
            "content_sha256": digests,
            "content_sha256_combined": combined,
        },
        "environment": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version},
    }
    meta.update(extra)
    side = out_path.with_name(out_path.name + ".json")
    side.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return side


def finish(out: Path, src: Path, args, counts, extra) -> int:
    summary, warns = check_output(out)
    extra = dict(extra)
    extra["checks"] = summary
    extra["warnings"] = warns
    side = write_sidecar(out, src, args, counts, extra, args.hash_source)
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and {side.name}")
    for k, v in summary.items():
        if k != "troute_layer_resolution":
            log(f"  check {k}: {v}")
    for w in warns:
        log(f"WARNING: {w}")
    return 1 if any("t-route" in w and "would read layer" in w for w in warns) else 0


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------
class Topology:
    """Flowpath graph from `flowpaths` (+ `nexus`) of an HFv4.0 gpkg."""

    def __init__(self, con: sqlite3.Connection, vpus: Optional[Sequence[str]] = None):
        fpc = set(columns(con, "flowpaths"))
        want = ["flowpath_id", "flowpath_toid", "total_dasqkm", "lake_id", "vpuid"]
        sel = ", ".join(c if c in fpc else f"NULL AS {c}" for c in want)
        where, params = "", ()
        if vpus:
            where = f" WHERE vpuid IN ({','.join('?' * len(vpus))})"
            params = tuple(vpus)
        self.toid: Dict[str, str] = {}
        self.area: Dict[str, Optional[float]] = {}
        self.lake: Dict[str, Optional[str]] = {}
        self.vpu: Dict[str, Optional[str]] = {}
        for fid, toid, area, lake, vpu in con.execute(f"SELECT {sel} FROM flowpaths{where}", params):
            self.toid[fid] = toid
            self.area[fid] = area
            self.lake[fid] = lake
            self.vpu[fid] = vpu
        nex_to: Dict[str, Optional[str]] = {}
        if has_table(con, "nexus"):
            nex_to = dict(con.execute("SELECT nexus_id, nexus_toid FROM nexus"))
        self.down: Dict[str, Optional[str]] = {}
        for f, t in self.toid.items():
            d = nex_to.get(t) if t else None
            self.down[f] = d if d in self.toid and d != f else None
        self.up: Dict[str, List[str]] = defaultdict(list)
        for f, d in self.down.items():
            if d is not None:
                self.up[d].append(f)
        for v in self.up.values():
            v.sort()

    def upstream_closure(self, outlets: Iterable[str]) -> List[str]:
        seen: Set[str] = set()
        q = deque()
        for o in outlets:
            if o not in self.toid:
                raise SystemExit(f"outlet {o} not found in flowpaths")
            if o not in seen:
                seen.add(o)
                q.append(o)
        while q:
            f = q.popleft()
            for u in self.up.get(f, ()):
                if u in seen:
                    raise SystemExit(f"flowpath graph is not a tree near {u}")
                seen.add(u)
                q.append(u)
        return sorted(seen)

    def topo_order(self) -> List[str]:
        """Headwaters first (Kahn). Raises on cycles."""
        indeg = {f: len(self.up.get(f, ())) for f in self.toid}
        q = deque(sorted(f for f, k in indeg.items() if k == 0))
        order: List[str] = []
        while q:
            f = q.popleft()
            order.append(f)
            d = self.down[f]
            if d is not None:
                indeg[d] -= 1
                if indeg[d] == 0:
                    q.append(d)
        if len(order) != len(self.toid):
            raise SystemExit(f"flowpath graph has a cycle ({len(self.toid) - len(order)} flowpaths unresolved)")
        return order


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
def cmd_vpu(args) -> int:
    src, out = Path(args.gpkg), Path(args.output)
    con = open_ro(src)
    layers = [n for n, _ in gpkg_layers(con)]
    if args.layers:
        missing = set(args.layers) - set(layers)
        if missing:
            raise SystemExit(f"layers not in source: {sorted(missing)}")
        layers = [n for n in layers if n in set(args.layers)]
    no_vpu = [n for n in layers if "vpuid" not in columns(con, n)]
    con.close()
    if no_vpu and not args.skip_layers_without_vpuid:
        raise SystemExit(
            f"layers without a vpuid column: {no_vpu}. Pass --skip-layers-without-vpuid to omit them."
        )
    ph = ",".join("?" * len(args.vpu))
    selectors = {n: Selector(f"vpuid IN ({ph})", args.vpu) for n in layers if n not in no_vpu}
    log(f"subsetting {src} to vpuid in {args.vpu}: {len(selectors)} layer(s)")
    counts = write_subset(src, out, selectors, overwrite=args.overwrite, spatial_index=not args.no_spatial_index)
    return finish(out, src, args, counts, {"selection": {"vpuid": args.vpu, "skipped_layers": no_vpu}})


def _flowline_stats(con, vpus) -> Dict[str, Tuple[int, int, int]]:
    """flowpath_id -> (n_flowlines, n_routeable, n_lake_flowlines)."""
    flc = set(columns(con, "flowlines")) if has_table(con, "flowlines") else set()
    if not {"flowpath_id", "routeable"} <= flc:
        return {}
    lake = "SUM(lake_id IS NOT NULL)" if "lake_id" in flc else "0"
    where, params = "", ()
    if vpus and "vpuid" in flc:
        where = f" WHERE vpuid IN ({','.join('?' * len(vpus))})"
        params = tuple(vpus)
    return {
        r[0]: (int(r[1]), int(r[2] or 0), int(r[3] or 0))
        for r in con.execute(
            f"SELECT flowpath_id, COUNT(*), SUM(routeable), {lake} FROM flowlines{where} GROUP BY flowpath_id",
            params,
        )
    }


def _gages(con) -> Dict[str, Tuple[List[str], List[str]]]:
    """flowpath_id -> (nwis site ids, rfc ids) from gage hydrolocations."""
    out: Dict[str, Tuple[List[str], List[str]]] = {}
    if not has_table(con, "hydrolocations"):
        return out
    for fp, ref in con.execute(
        "SELECT flowpath_id, hl_reference FROM hydrolocations WHERE hl_class='gage' AND flowpath_id IS NOT NULL"
    ):
        nw, rf = out.setdefault(fp, ([], []))
        for x in split_refs(ref):
            if x.startswith("nwis-"):
                nw.append(x[5:])
            elif x.startswith("rfc-"):
                rf.append(x[4:])
    return out


def cmd_candidates(args) -> int:
    src = Path(args.gpkg)
    con = open_ro(src)
    t0 = time.time()
    topo = Topology(con, args.vpu)
    fls = _flowline_stats(con, args.vpu)
    gages = _gages(con)
    lakes_with_params = {r[0] for r in con.execute("SELECT lake_id FROM lakes")} if has_table(con, "lakes") else set()
    n_src_fp = con.execute("SELECT count(*) FROM flowpaths").fetchone()[0]
    con.close()
    bytes_per_fp = src.stat().st_size / max(n_src_fp, 1)
    log(f"loaded {len(topo.toid):,d} flowpaths, {len(gages):,d} gauged flowpaths in {time.time() - t0:.1f} s")

    order = topo.topo_order()
    keys = ("n_fp", "n_fl", "n_nonroute_fl", "n_gage_fp", "n_lakes", "n_conf",
            "n_gage_no_routeable", "n_gage_with_rfc", "n_gage_below_lake")
    acc = {f: dict.fromkeys(keys, 0) for f in topo.toid}
    chain = dict.fromkeys(topo.toid, 0)
    lakes_up = dict.fromkeys(topo.toid, 0)  # lake outlets strictly upstream, incl. self
    for f in order:
        a = acc[f]
        nfl, nrt, _ = fls.get(f, (0, 0, 0))
        nw, rf = gages.get(f, ([], []))
        gauged = bool(nw)
        lk = topo.lake.get(f)
        d = topo.down[f]
        lake_outlet = lk is not None and lk in lakes_with_params and (d is None or topo.lake.get(d) != lk)
        a["n_fp"] += 1
        a["n_fl"] += nfl
        a["n_nonroute_fl"] += nfl - nrt
        a["n_gage_fp"] += gauged
        a["n_lakes"] += lake_outlet
        a["n_conf"] += len(topo.up.get(f, ())) >= 2
        a["n_gage_no_routeable"] += gauged and nfl > 0 and nrt == 0
        a["n_gage_with_rfc"] += gauged and bool(rf)
        lakes_up[f] += lake_outlet
        a["n_gage_below_lake"] += gauged and (lakes_up[f] - lake_outlet) > 0
        chain[f] += gauged
        if d is not None:
            ad = acc[d]
            for k in keys:
                ad[k] += a[k]
            chain[d] = max(chain[d], chain[f])
            lakes_up[d] += lakes_up[f]

    rows = []
    for f in topo.toid:
        a = acc[f]
        if topo.lake.get(f) is not None:
            continue  # outlet inside a lake would cut the lake
        if not (a["n_gage_fp"] >= args.min_gages and chain[f] >= 2 and a["n_lakes"] >= 1
                and a["n_conf"] >= 1 and a["n_nonroute_fl"] >= 1 and a["n_fp"] <= args.max_flowpaths):
            continue
        toid = topo.toid[f] or ""
        rows.append(dict(
            outlet=f, toid=toid, tnx=toid.startswith("tnx-"), vpuid=topo.vpu.get(f),
            total_dasqkm=topo.area.get(f), gage_chain=chain[f], est_mb=round(a["n_fp"] * bytes_per_fp / 1e6, 1), **a,
        ))
    rows.sort(key=lambda r: (not r["tnx"], r["n_gage_below_lake"] == 0, r["n_gage_no_routeable"] == 0,
                             r["n_fp"], r["outlet"]))
    log(f"{len(rows):,d} candidate outlet(s) meet the criteria")

    cols = ["outlet", "toid", "tnx", "vpuid", "total_dasqkm", "n_fp", "n_fl", "n_nonroute_fl", "n_gage_fp",
            "gage_chain", "n_lakes", "n_conf", "n_gage_below_lake", "n_gage_no_routeable", "n_gage_with_rfc", "est_mb"]
    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        log(f"wrote {args.csv}")
    print("\t".join(cols))
    for r in rows[: args.top]:
        print("\t".join("" if r[c] is None else (f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c])) for c in cols))
    for r in rows[: args.detail]:
        basin = topo.upstream_closure([r["outlet"]])
        g = [(f, gages[f][0], gages[f][1]) for f in basin if f in gages and gages[f][0]]
        lk = sorted({topo.lake[f] for f in basin if topo.lake.get(f) is not None})
        print(f"\n# {r['outlet']} -> {r['toid']}: {len(basin)} flowpaths")
        for f, nw, rf in g:
            print(f"#   gage {f}: nwis={','.join(nw)}" + (f" rfc={','.join(rf)}" if rf else ""))
        print(f"#   lakes: {', '.join(lk)}")
    print(
        "\n# columns: n_fp flowpaths in basin; n_nonroute_fl non-routeable flowlines; n_gage_fp gauged (nwis) "
        "flowpaths; gage_chain longest run of nested gauged flowpaths (>=2 means gages in series); n_lakes lake "
        "outlets with a lakes row; n_conf confluences; n_gage_below_lake gauged flowpaths with a lake upstream; "
        "n_gage_no_routeable gauged flowpaths with no routeable flowline (exercises a flowline-mode drop); "
        "n_gage_with_rfc gauged flowpaths also carrying an rfc- id; est_mb from source bytes per flowpath."
    )
    return 0


def cmd_upstream(args) -> int:
    src, out = Path(args.gpkg), Path(args.output)
    con = open_ro(src)
    topo = Topology(con)
    basin = topo.upstream_closure(args.outlet)
    for o in args.outlet:
        if topo.lake.get(o) is not None:
            log(f"WARNING: outlet {o} is inside lake {topo.lake[o]}; the lake will be cut")
    sel_fp = set(basin)
    layers = [n for n, _ in gpkg_layers(con)]
    sel_fl: Set[int] = set()
    sel_nex: Set[str] = {topo.toid[f] for f in basin if topo.toid[f]}
    sel_lake: Set[str] = {topo.lake[f] for f in basin if topo.lake.get(f) is not None}
    if "flowlines" in layers:
        flc = set(columns(con, "flowlines"))
        con.execute("CREATE TEMP TABLE b (id PRIMARY KEY) WITHOUT ROWID")
        con.executemany("INSERT INTO temp.b VALUES (?)", ((f,) for f in basin))
        lake_col = "lake_id" if "lake_id" in flc else "NULL"
        toid_col = "flowpath_toid" if "flowpath_toid" in flc else "NULL"
        for fl, toid, lk in con.execute(
            f"SELECT flowline_id, {toid_col}, {lake_col} FROM flowlines WHERE flowpath_id IN (SELECT id FROM temp.b)"
        ):
            sel_fl.add(fl)
            if toid:
                sel_nex.add(toid)
            if lk is not None:
                sel_lake.add(lk)
    selectors: Dict[str, Selector] = {}
    skipped = []
    for n in layers:
        c = set(columns(con, n))
        if n == "nexus" and "nexus_id" in c:
            selectors[n] = Selector("nexus_id IN (SELECT id FROM temp.sel_nex)")
        elif n == "lakes" and "lake_id" in c:
            extra = " OR flowpath_id IN (SELECT id FROM temp.sel_fp)" if "flowpath_id" in c else ""
            selectors[n] = Selector("lake_id IN (SELECT id FROM temp.sel_lake)" + extra)
        elif "flowpath_id" in c:
            selectors[n] = Selector("flowpath_id IN (SELECT id FROM temp.sel_fp)")
        elif "flowline_id" in c:
            selectors[n] = Selector("flowline_id IN (SELECT id FROM temp.sel_fl)")
        else:
            skipped.append(n)
    con.close()
    for n in skipped:
        log(f"WARNING: layer {n} has no flowpath_id/flowline_id/nexus_id/lake_id key; not copied")
    log(f"subsetting {src} upstream of {args.outlet}: {len(basin)} flowpaths, {len(sel_fl)} flowlines")
    counts = write_subset(
        src, out, selectors,
        temp_ids={"sel_fp": sel_fp, "sel_fl": sel_fl, "sel_nex": sel_nex, "sel_lake": sel_lake},
        overwrite=args.overwrite, spatial_index=not args.no_spatial_index,
    )
    extra = {"selection": {"outlets": args.outlet, "n_flowpaths": len(basin), "n_flowlines": len(sel_fl),
                           "skipped_layers": skipped}}
    return finish(out, src, args, counts, extra)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=TOOL_NAME, description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {SCRIPT_VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common_out(sp):
        sp.add_argument("-o", "--output", required=True, help="output .gpkg")
        sp.add_argument("--overwrite", action="store_true", help="replace an existing output")
        sp.add_argument("--no-spatial-index", action="store_true", help="do not copy R-tree indexes")
        sp.add_argument("--hash-source", action="store_true",
                        help="record the source file's sha256 in the sidecar (reads the whole file)")

    s = sub.add_parser("vpu", help="copy all layers filtered by vpuid")
    s.add_argument("gpkg", help="source HFv4.0 GeoPackage")
    s.add_argument("--vpu", action="append", required=True, help="vpuid to keep (repeatable), e.g. 01, 03N, 10L")
    s.add_argument("--layers", nargs="+", help="only these layers (default: all vector/attribute layers)")
    s.add_argument("--skip-layers-without-vpuid", action="store_true")
    common_out(s)
    s.set_defaults(func=cmd_vpu)

    s = sub.add_parser("candidates", help="rank fixture outlet candidates")
    s.add_argument("gpkg", help="source HFv4.0 GeoPackage (CONUS or a VPU subset)")
    s.add_argument("--vpu", action="append", help="restrict to these vpuids (repeatable)")
    s.add_argument("--min-gages", type=int, default=3, help="minimum gauged (nwis) flowpaths (default 3)")
    s.add_argument("--max-flowpaths", type=int, default=400, help="maximum basin size in flowpaths (default 400)")
    s.add_argument("--top", type=int, default=20, help="rows to print (default 20)")
    s.add_argument("--detail", type=int, default=5, help="candidates to list gages/lakes for (default 5)")
    s.add_argument("--csv", help="write all candidates to this CSV")
    s.set_defaults(func=cmd_candidates)

    s = sub.add_parser("upstream", help="copy the basin upstream of outlet flowpath(s)")
    s.add_argument("gpkg", help="source HFv4.0 GeoPackage")
    s.add_argument("--outlet", action="append", required=True, help="outlet flowpath_id, e.g. fp-12345 (repeatable)")
    common_out(s)
    s.set_defaults(func=cmd_upstream)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
