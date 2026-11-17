"""HYFeatures (HFv4.0) network adapter for area-scaled streamflow DA.

Extracts the inputs of `build_upstream_trees`
from a built `HYFeaturesNetwork`, in flowpath
or flowline (`routeable`) mode:

* `rconn`: the network's reverse connectivity over the *routed* graph
  (waterbodies collapsed to one node per lake when the network is broken at
  waterbodies), reused as-is;
* `drainage_km2`: hydrofabric `total_dasqkm` of the routed layer, used for both
  A_s and A_o (`poi_da_*` is reported, never used);
* `length_km`: routed segment length (`dx`, m) / 1000;
* `gage_segments`: from the network's own gage crosswalk (`network.gages`, the
  one nudging uses), restricted to USGS (`nwis`) sites by default;
* `site_vpu`: `vpuid` of each gage's routed segment;
* stops: every waterbody segment and lake node, plus diffusive-domain segments
  and any caller-supplied segments.

Gages that cannot be assimilated are dropped with a logged reason
(`ADAPTER_DROP_REASONS`, then the core's
`DROP_REASONS`).

The adapter only reads these network attributes, so tests can pass a light
stand-in: `reverse_network`, `dataframe` (`dx` column, index = segment),
`segment_attributes`, `gage_hydrolocations`, `gages`,
`waterbody_connections`, `compute_parameters`, `supernetwork_parameters`
and `diffusive_network_data`.

References: the method follows the NGWPC t-route implementation of
area-scaled streamflow DA; this adapter is written against
`HYFeaturesNetwork` and HFv4.0 field names.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from troute.scaling_da.forest import TreeBuildResult, build_upstream_trees
from troute.scaling_da.theta import DEFAULT_THETA

__all__ = [
    "ADAPTER_DROP_REASONS",
    "DEFAULT_MAX_REACH_KM",
    "DEFAULT_POI_RATIO_TOLERANCE",
    "ArrayMapping",
    "HYFeaturesMappings",
    "HYFeaturesTreeBuild",
    "build_hyfeatures_trees",
    "hyfeatures_mappings",
    "network_mode",
    "results_seg_to_row",
    "routed_segment_ids",
]

LOG = logging.getLogger(__name__)

# Divisions by zero and 0/0 are handled explicitly after the division.
_IGNORE_FP = {"divide": "ignore", "invalid": "ignore"}

# Default upstream distance cap (km), as in NGWPC's `max_reach_km`.
DEFAULT_MAX_REACH_KM = 200.0
# Relative tolerance for flagging gage drainage-area disagreement (report only).
DEFAULT_POI_RATIO_TOLERANCE = 0.25

# Reasons the adapter drops a gage before tree building.
ADAPTER_DROP_REASONS = {
    "not_in_crosswalk": (
        "gage hydrolocation has no segment in the network's gage crosswalk (e.g. its flowpath has "
        "no routed flowline, or another id on the same segment replaced it)"
    ),
    "not_routed": "gage segment is not in the MC routing graph (e.g. diffusive domain)",
}


class ArrayMapping(_MappingABC):
    """Read-only `int -> float` mapping on sorted NumPy arrays.

    Much smaller than a dict for millions of segments; `get` is a binary
    search. Duplicate keys keep the first value.
    """

    __slots__ = ("_keys", "_values")

    def __init__(self, keys: Iterable[Any], values: Iterable[Any]):
        k = np.asarray(keys, dtype=np.int64).ravel()
        v = np.asarray(values, dtype=np.float64).ravel()
        if k.shape != v.shape:
            raise ValueError(f"keys and values differ in length ({k.size} vs {v.size})")
        order = np.argsort(k, kind="stable")
        k, v = k[order], v[order]
        if k.size:
            first = np.concatenate(([True], k[1:] != k[:-1]))
            k, v = k[first], v[first]
        self._keys = k
        self._values = v

    def _index(self, key: Any) -> int:
        try:
            key = int(key)
        except (TypeError, ValueError, OverflowError):
            return -1
        if not (-(2**63) <= key < 2**63):
            return -1
        i = int(np.searchsorted(self._keys, key))
        return i if i < self._keys.size and int(self._keys[i]) == key else -1

    def __getitem__(self, key: Any) -> float:
        i = self._index(key)
        if i < 0:
            raise KeyError(key)
        return float(self._values[i])

    def get(self, key: Any, default: Optional[float] = None) -> Optional[float]:
        i = self._index(key)
        return default if i < 0 else float(self._values[i])

    def __contains__(self, key: object) -> bool:
        return self._index(key) >= 0

    def __iter__(self) -> Iterator[int]:
        return iter(self._keys.tolist())

    def __len__(self) -> int:
        return int(self._keys.size)

    def lookup(self, keys: Iterable[Any]) -> np.ndarray:
        """Vectorised lookup; NaN where a key is missing."""
        q = np.asarray(keys, dtype=np.int64).ravel()
        out = np.full(q.shape, np.nan)
        if self._keys.size and q.size:
            i = np.minimum(np.searchsorted(self._keys, q), self._keys.size - 1)
            hit = self._keys[i] == q
            out[hit] = self._values[i[hit]]
        return out


@dataclass(frozen=True)
class HYFeaturesMappings:
    """Inputs for tree building, extracted from a HYFeatures network.

    Attributes
    ----------
    mode : str
        `"flowpath"` or `"flowline"` (from the network's `key` column).
    rconn : Mapping[int, Sequence[int]]
        The network's reverse connectivity over the routed graph.
    drainage_km2, length_km : ArrayMapping
        `total_dasqkm` and routed length (km) per routed segment.
    gage_segments : dict[str, int]
        Assimilated sites, ordered by (segment, site).
    site_vpu : dict[str, str]
        `vpuid` of each site in `gage_segments`.
    lake_segments : frozenset[int]
        Lake member segments and lake nodes (stops).
    extra_stop_segs : frozenset[int]
        Diffusive-domain segments plus caller-supplied stops.
    dropped : dict[str, str]
        `site -> reason` (`ADAPTER_DROP_REASONS`).
    gage_report : pandas.DataFrame
        One row per gage site of an assimilated type: `site, type, flowpath_id,
        segment, status, vpuid, area_model_sqkm, poi_da_ref_sqkm,
        poi_da_ngen_sqkm, poi_da_ratio, poi_da_flag, model_to_ref, flagged`.
        `status` is `"mapped"` or a drop reason. `flagged` marks
        drainage-area disagreement; it is reported, never used to exclude.
    excluded_types : dict[str, int]
        Gage sites of other types (e.g. `rfc`), counted, not assimilated.
    n_area_inversions_network : int
        Routed links whose upstream `total_dasqkm` exceeds the downstream one.
    """

    mode: str
    rconn: Mapping[int, Sequence[int]]
    drainage_km2: ArrayMapping
    length_km: ArrayMapping
    gage_segments: Dict[str, int]
    site_vpu: Dict[str, str]
    lake_segments: FrozenSet[int]
    extra_stop_segs: FrozenSet[int]
    dropped: Dict[str, str]
    gage_report: pd.DataFrame
    excluded_types: Dict[str, int]
    n_area_inversions_network: int


@dataclass(frozen=True)
class HYFeaturesTreeBuild:
    """Result of `build_hyfeatures_trees`.

    `dropped` merges adapter and core drop reasons; `gage_report` adds the
    final `status` and `n_tree_segments` per site.
    """

    mappings: HYFeaturesMappings
    result: TreeBuildResult
    dropped: Dict[str, str]
    gage_report: pd.DataFrame


def network_mode(network: Any) -> str:
    """`"flowline"` if the routed key column is a flowline id, else `"flowpath"`."""
    cols = (getattr(network, "supernetwork_parameters", None) or {}).get("columns") or {}
    return "flowline" if "line" in str(cols.get("key", "")) else "flowpath"


def routed_segment_ids(network: Any) -> np.ndarray:
    """Sorted ids of every node in the routed graph (segments and lake nodes)."""
    return np.asarray(sorted(int(s) for s in network.reverse_network.keys()), dtype=np.int64)


def results_seg_to_row(index: Iterable[Any]) -> Dict[int, int]:
    """`segment id -> row` for a routed-results index (e.g. `flowveldepth.index`)."""
    out: Dict[int, int] = {}
    for row, s in enumerate(index):
        s = int(s)
        if s in out:
            raise ValueError(f"results index has duplicate segment id {s}")
        out[s] = row
    return out


def _diffusive_segments(network: Any) -> FrozenSet[int]:
    data = getattr(network, "diffusive_network_data", None) or {}
    segs = set()
    for tw, d in data.items():
        segs.add(int(tw))
        segs.update(int(s) for s in (d or {}).get("mainstem_segs", ()))
    return frozenset(segs)


def _outside(x: pd.Series, tol: float) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype("float64")
    lim = math.log1p(tol)
    with np.errstate(**_IGNORE_FP):
        dev = np.abs(np.log(x.where(x > 0)))
    return (dev > lim).fillna(False).astype(bool)


def _area_inversions(rconn: Mapping[int, Sequence[int]], area: ArrayMapping) -> int:
    down, up = [], []
    for d, ups in rconn.items():
        for u in ups:
            down.append(d)
            up.append(u)
    if not down:
        return 0
    a_d, a_u = area.lookup(down), area.lookup(up)
    ok = np.isfinite(a_d) & np.isfinite(a_u)
    return int(np.count_nonzero(a_u[ok] > a_d[ok]))


def hyfeatures_mappings(
    network: Any,
    *,
    site_types: Sequence[str] = ("nwis",),
    extra_stop_segs: Iterable[int] = (),
    poi_ratio_tolerance: float = DEFAULT_POI_RATIO_TOLERANCE,
) -> HYFeaturesMappings:
    """Extract tree-building inputs from a built HYFeatures network.

    Parameters
    ----------
    network : HYFeaturesNetwork
        Built network (or a stand-in with the attributes listed in the module docstring).
    site_types : Sequence[str], optional
        `hl_reference` prefixes to assimilate. Default `("nwis",)` (USGS).
        Other types are counted in `excluded_types`; they get no tree and do
        not stop other trees.
    extra_stop_segs : Iterable[int], optional
        Additional stop segments (e.g. gages assimilated by another method).
    poi_ratio_tolerance : float, optional
        Relative tolerance for `flagged` in the gage report.

    Raises
    ------
    NotImplementedError
        If the network uses `compute_nodes` (areas per compute node are not defined).
    ValueError
        If `total_dasqkm` or the gage hydrolocations are not available.
    """
    cp = getattr(network, "compute_parameters", None) or {}
    if cp.get("compute_nodes", False):
        raise NotImplementedError(
            "area-scaled streamflow DA does not support compute_nodes yet: "
            "drainage areas per compute node are not defined"
        )
    seg_attrs = getattr(network, "segment_attributes", None)
    if seg_attrs is None or seg_attrs.empty or "total_dasqkm" not in seg_attrs.columns:
        raise ValueError(
            "area-scaled streamflow DA needs total_dasqkm per routed segment "
            "(network.segment_attributes); the hydrofabric layer being routed has none"
        )
    hl = getattr(network, "gage_hydrolocations", None)
    if hl is None or hl.empty:
        raise ValueError(
            "area-scaled streamflow DA needs gage hydrolocations (network.gage_hydrolocations); "
            "they are read only when streamflow DA is enabled"
        )

    rconn = network.reverse_network
    area = ArrayMapping(seg_attrs.index.values, seg_attrs["total_dasqkm"].values)
    df = network.dataframe
    length = ArrayMapping(df.index.values, np.asarray(df["dx"].values, dtype=np.float64) / 1000.0)
    wb = getattr(network, "waterbody_connections", None) or {}
    lake_segments = frozenset(int(k) for k in wb) | frozenset(int(v) for v in wb.values())
    diffusive = _diffusive_segments(network)
    extra = frozenset(int(s) for s in extra_stop_segs) | diffusive

    gages_attr = getattr(network, "gages", None) or {}
    crosswalk = gages_attr.get("gages") or {}
    site_to_seg: Dict[str, int] = {}
    for seg, site in crosswalk.items():
        site_to_seg.setdefault(str(site), int(seg))

    hl = hl.copy()
    hl["site"] = hl["site"].astype(str)
    types = tuple(str(t) for t in site_types)
    keep = hl["type"].isin(types)
    other = hl.loc[~keep].drop_duplicates(["type", "site"])
    excluded_types = {str(k): int(v) for k, v in other["type"].value_counts().sort_index().items()}
    sites = hl.loc[keep].sort_values("site", kind="stable").drop_duplicates("site", keep="first")

    has_vpu = "vpuid" in seg_attrs.columns
    mapped: list = []
    dropped: Dict[str, str] = {}
    status, segcol = [], []
    for site in sites["site"]:
        seg = site_to_seg.get(site)
        segcol.append(seg)
        if seg is None:
            dropped[site] = "not_in_crosswalk"
        elif seg in lake_segments:
            mapped.append((seg, site))  # core drops it as gage_on_waterbody
        elif seg in diffusive or seg not in rconn:
            dropped[site] = "not_routed"
        else:
            mapped.append((seg, site))
        status.append(dropped.get(site, "mapped"))
    mapped.sort()
    gage_segments = {site: seg for seg, site in mapped}

    report = pd.DataFrame(
        {
            "site": sites["site"].values,
            "type": sites["type"].values,
            "flowpath_id": sites["flowpath_id"].values if "flowpath_id" in sites else None,
            "segment": pd.array(segcol, dtype="Int64"),
            "status": status,
        }
    )
    seg_vals = report["segment"].astype("float64").fillna(-1).astype("int64").values
    report["area_model_sqkm"] = area.lookup(seg_vals)
    seg_vpu = (
        seg_attrs["vpuid"].astype(object).reindex(seg_vals).values if has_vpu else np.full(len(report), None)
    )
    hl_vpu = sites["vpuid"].astype(object).values if "vpuid" in sites else np.full(len(report), None)
    report["vpuid"] = [sv if isinstance(sv, str) else hv for sv, hv in zip(seg_vpu, hl_vpu)]
    for c in ("poi_da_ref_sqkm", "poi_da_ngen_sqkm", "poi_da_ratio", "poi_da_flag"):
        report[c] = pd.to_numeric(sites[c], errors="coerce").values if c in sites else np.nan
    with np.errstate(**_IGNORE_FP):
        report["model_to_ref"] = report["area_model_sqkm"] / report["poi_da_ref_sqkm"].where(
            report["poi_da_ref_sqkm"] > 0
        )
    report["flagged"] = (
        (report["poi_da_flag"].fillna(0) != 0)
        | _outside(report["poi_da_ratio"], poi_ratio_tolerance)
        | _outside(report["model_to_ref"], poi_ratio_tolerance)
    )
    site_vpu = {
        s: v for s, v in zip(report["site"], report["vpuid"]) if s in gage_segments and isinstance(v, str)
    }

    n_inv = _area_inversions(rconn, area)
    mappings = HYFeaturesMappings(
        mode=network_mode(network),
        rconn=rconn,
        drainage_km2=area,
        length_km=length,
        gage_segments=gage_segments,
        site_vpu=site_vpu,
        lake_segments=lake_segments,
        extra_stop_segs=extra,
        dropped=dropped,
        gage_report=report,
        excluded_types=excluded_types,
        n_area_inversions_network=n_inv,
    )
    _log_mappings(mappings, types)
    return mappings


def _log_mappings(m: HYFeaturesMappings, types: Sequence[str]) -> None:
    LOG.info(
        "scaling DA (%s mode): %d %s gage site(s), %d mapped to routed segments, %d dropped; "
        "other gage types not assimilated: %s",
        m.mode, len(m.gage_report), "/".join(types), len(m.gage_segments), len(m.dropped),
        m.excluded_types or "none",
    )
    by_reason: Dict[str, list] = {}
    for site, reason in m.dropped.items():
        by_reason.setdefault(reason, []).append(site)
    for reason, sites in sorted(by_reason.items()):
        LOG.warning(
            "scaling DA: dropped %d gage(s), %s: %s%s",
            len(sites), ADAPTER_DROP_REASONS[reason], sites[:10], " ..." if len(sites) > 10 else "",
        )
    n_flag = int(m.gage_report["flagged"].sum())
    if n_flag:
        LOG.warning(
            "scaling DA: %d gage(s) disagree with the hydrofabric drainage area "
            "(poi_da_flag, or poi_da_ratio/model-to-reference ratio outside the tolerance); "
            "reported only, still assimilated: %s",
            n_flag, m.gage_report.loc[m.gage_report["flagged"], "site"].tolist()[:10],
        )
    if m.n_area_inversions_network:
        LOG.info(
            "scaling DA: %d routed link(s) have upstream total_dasqkm > downstream",
            m.n_area_inversions_network,
        )


def build_hyfeatures_trees(
    network: Any,
    *,
    site_theta: Optional[Mapping[str, float]] = None,
    default_theta: float = DEFAULT_THETA,
    max_reach_km: Optional[float] = DEFAULT_MAX_REACH_KM,
    gage_drainage_km2: Optional[Mapping[str, float]] = None,
    site_types: Sequence[str] = ("nwis",),
    extra_stop_segs: Iterable[int] = (),
    poi_ratio_tolerance: float = DEFAULT_POI_RATIO_TOLERANCE,
) -> HYFeaturesTreeBuild:
    """Build the disjoint upstream gage trees of a HYFeatures network.

    Stops at every waterbody, every other assimilated gage, diffusive-domain
    segments and `extra_stop_segs`. `max_reach_km=None` disables the
    distance cap. See `hyfeatures_mappings` for the other arguments.
    """
    m = hyfeatures_mappings(
        network,
        site_types=site_types,
        extra_stop_segs=extra_stop_segs,
        poi_ratio_tolerance=poi_ratio_tolerance,
    )
    res = build_upstream_trees(
        m.rconn,
        m.gage_segments,
        m.drainage_km2,
        site_theta=site_theta,
        default_theta=default_theta,
        lake_segments=m.lake_segments,
        extra_stop_segs=m.extra_stop_segs,
        stop_at_lakes=True,
        gage_drainage_km2=gage_drainage_km2,
        length_km=m.length_km if max_reach_km is not None else None,
        max_reach_km=max_reach_km,
    )
    dropped = dict(m.dropped)
    dropped.update(res.dropped)
    report = m.gage_report.copy()
    report["status"] = [dropped.get(s, "tree") for s in report["site"]]
    report["n_tree_segments"] = [
        res.trees[s].n_nodes if s in res.trees else 0 for s in report["site"]
    ]
    return HYFeaturesTreeBuild(mappings=m, result=res, dropped=dropped, gage_report=report)
