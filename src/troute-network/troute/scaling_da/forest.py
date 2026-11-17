"""Build disjoint upstream trees for a set of assimilated gages.

Network-agnostic: callers supply reverse connectivity, `gage -> segment`,
per-segment drainage area (and optionally length), and the waterbody stop set.
The HYFeatures/HFv4 adapter that extracts these from a t-route network is
`troute.scaling_da.hyfeatures`.

References: the method follows the NGWPC t-route implementation of
area-scaled streamflow DA; science reference Ogden & Dawdy (2003),
J. Hydrol. Eng. 8(2):64-73 (see `troute.scaling_da`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from troute.scaling_da.tree import UpstreamTree, trace_upstream_tree, cap_tree_by_distance
from troute.scaling_da.theta import DEFAULT_THETA, check_theta

__all__ = ["DROP_REASONS", "TreeBuildResult", "build_upstream_trees"]

LOG = logging.getLogger(__name__)

# Reasons a gage gets no tree, as reported in `TreeBuildResult.dropped`.
DROP_REASONS = {
    "colocated": "another gage already assimilates this segment (first one wins)",
    "gage_on_waterbody": "gage segment is a waterbody segment (reservoir DA's domain)",
    "gage_area_invalid": "gage drainage area missing, non-finite or <= 0",
    "nonfinite_area": "a tree segment has a missing/non-finite drainage area",
}


@dataclass(frozen=True)
class TreeBuildResult:
    """Output of `build_upstream_trees`.

    Attributes
    ----------
    trees : dict[str, UpstreamTree]
        One tree per kept site, in input order. Pairwise disjoint.
    dropped : dict[str, str]
        `site -> reason key` (see `DROP_REASONS`).
    n_capped : dict[str, int]
        Nodes removed per site by the distance cap (sites with > 0 only).
    """

    trees: Dict[str, UpstreamTree]
    dropped: Dict[str, str]
    n_capped: Dict[str, int]


SiteSegments = Mapping[str, int]
"""`site id -> routed segment id` of the gages to build trees for."""


def build_upstream_trees(
    rconn: Mapping[int, Sequence[int]],
    gage_segments: SiteSegments,
    drainage_km2: Mapping[int, float],
    *,
    site_theta: Optional[Mapping[str, float]] = None,
    default_theta: float = DEFAULT_THETA,
    lake_segments: Iterable[int] = (),
    extra_stop_segs: Iterable[int] = (),
    stop_at_lakes: bool = True,
    gage_drainage_km2: Optional[Mapping[str, float]] = None,
    length_km: Optional[Mapping[int, float]] = None,
    max_reach_km: Optional[float] = None,
) -> TreeBuildResult:
    """Build one upstream tree per assimilated gage.

    Stop rules: every branch ends at (excludes) the segment of any *other* gage
    in `gage_segments`, any segment in `extra_stop_segs`, and, if
    `stop_at_lakes`, any segment in `lake_segments`. Because every
    tree root is a stop for every other tree, the trees are disjoint.

    Held-out (evaluation) gages: leave them out of `gage_segments` and out of
    `extra_stop_segs`. They then get no tree and corrections from the next
    gage downstream pass through them.

    Parameters
    ----------
    rconn : Mapping[int, Sequence[int]]
        For each segment, the segments that flow directly into it.
    gage_segments : Mapping[str, int]
        `site -> segment` for the gages whose innovations are spread. Input
        order decides which of two co-located gages wins.
    drainage_km2 : Mapping[int, float]
        Total drainage area per segment (km²).
    site_theta : Mapping[str, float], optional
        Resolved exponent per site (see `resolve_theta`);
        sites not present use `default_theta`.
    default_theta : float, optional
        Default exponent (finite, > 0).
    lake_segments : Iterable[int], optional
        Waterbody segments (stops when `stop_at_lakes`).
    extra_stop_segs : Iterable[int], optional
        Additional stop segments (e.g. gages assimilated by another method).
    stop_at_lakes : bool, optional
        Default True.
    gage_drainage_km2 : Mapping[str, float], optional
        Per-site override for A_o. Default: the gage segment's `drainage_km2`.
    length_km, max_reach_km : optional
        If both are given, each tree is capped with
        `cap_tree_by_distance`.

    Returns
    -------
    TreeBuildResult
    """
    default_theta = check_theta(default_theta, "theta default")
    if (length_km is None) != (max_reach_km is None):
        raise ValueError("length_km and max_reach_km must be given together")
    sites_segs = [(str(s), int(g)) for s, g in gage_segments.items()]
    all_gage_segs = frozenset(g for _, g in sites_segs)
    waterbodies = frozenset(int(s) for s in lake_segments)
    base_stop = set(int(s) for s in extra_stop_segs) | set(all_gage_segs)
    if stop_at_lakes:
        base_stop |= waterbodies
    site_theta = site_theta or {}
    area_override = {str(k): v for k, v in (gage_drainage_km2 or {}).items()}

    trees: Dict[str, UpstreamTree] = {}
    dropped: Dict[str, str] = {}
    n_capped: Dict[str, int] = {}
    claimed = set()
    for site, seg in sites_segs:
        if seg in claimed:
            dropped[site] = "colocated"
            continue
        if stop_at_lakes and seg in waterbodies:
            dropped[site] = "gage_on_waterbody"
            continue
        a_o = area_override.get(site, drainage_km2.get(seg))
        if a_o is None or not (math.isfinite(float(a_o)) and float(a_o) > 0):
            dropped[site] = "gage_area_invalid"
            continue
        tree = trace_upstream_tree(
            seg,
            rconn,
            drainage_km2,
            base_stop - {seg},
            theta=check_theta(site_theta.get(site, default_theta), f"theta[{site}]"),
            site=site,
            gage_drainage_km2=float(a_o),
        )
        if length_km is not None:
            tree, k = cap_tree_by_distance(tree, length_km, float(max_reach_km))
            if k:
                n_capped[site] = k
        # A NaN area would put NaN into discharge (and, at hand-off, into state).
        if not np.all(np.isfinite(tree.drainage_km2)):
            dropped[site] = "nonfinite_area"
            continue
        trees[site] = tree
        claimed.add(seg)

    _check_disjoint(trees)
    _log_build(trees, dropped, n_capped)
    return TreeBuildResult(trees=trees, dropped=dropped, n_capped=n_capped)


def _check_disjoint(trees: Mapping[str, UpstreamTree]) -> None:
    if not trees:
        return
    allsegs = np.concatenate([t.seg_ids for t in trees.values()])
    if np.unique(allsegs).size != allsegs.size:
        # Unreachable with the stop rules above; guards future edits.
        raise RuntimeError("scaling DA: gage trees overlap; corrections would double count")


def _log_build(trees, dropped, n_capped) -> None:
    n_nodes = sum(t.n_nodes for t in trees.values())
    LOG.info(
        "scaling DA: built %d gage tree(s), %d segment(s) total, %d gage(s) dropped",
        len(trees), n_nodes, len(dropped),
    )
    by_reason: Dict[str, list] = {}
    for site, reason in dropped.items():
        by_reason.setdefault(reason, []).append(site)
    for reason, sites in by_reason.items():
        LOG.warning(
            "scaling DA: dropped %d gage(s), %s: %s%s",
            len(sites), DROP_REASONS[reason], sites[:10], " ..." if len(sites) > 10 else "",
        )
    if n_capped:
        LOG.info(
            "scaling DA: distance cap removed %d segment(s) from %d tree(s)",
            sum(n_capped.values()), len(n_capped),
        )
