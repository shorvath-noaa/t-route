"""Per-gage upstream tree for area-scaled streamflow DA.

A `UpstreamTree` is a gage segment plus every segment upstream of it, up to
(not including) the first stop segment on each branch. It is stored as flat
NumPy arrays in breadth-first order, so every node's parent precedes it.

Segment ids are integers (t-route's routed segment ids). The functions here
only need a reverse-connectivity mapping, a per-segment area lookup and a stop
set, so they are independent of the hydrofabric.

References: the method follows the NGWPC t-route implementation of
area-scaled streamflow DA; science reference Ogden & Dawdy (2003),
J. Hydrol. Eng. 8(2):64-73 (see `troute.scaling_da`).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "UpstreamTree",
    "trace_upstream_tree",
    "cap_tree_by_distance",
    "tree_distance_km",
]


def _csr(groups: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten per-node lists into CSR (`ptr` of length n+1, `flat`)."""
    ptr = np.zeros(len(groups) + 1, dtype=np.int64)
    flat: List[int] = []
    for i, g in enumerate(groups):
        flat.extend(int(x) for x in g)
        ptr[i + 1] = len(flat)
    return ptr, np.asarray(flat, dtype=np.int64)


@dataclass(frozen=True)
class UpstreamTree:
    """Upstream tree of one gage, in breadth-first order (gage first).

    Attributes
    ----------
    site : str
        Gage id (e.g. USGS site number). Informational.
    seg_ids : ndarray[int64], shape (n,)
        Segment ids; `seg_ids[0]` is the gage segment.
    parent : ndarray[int64], shape (n,)
        Local index of each node's downstream neighbour; `-1` for the gage.
        `parent[i] < i` for all `i > 0`.
    drainage_km2 : ndarray[float64], shape (n,)
        Total drainage area per node. `drainage_km2[0]` is the gage area
        A_o actually used by the spread (the modeled area, or an
        override supplied at build time).
    is_junction_step : ndarray[bool], shape (n,)
        True when the node's parent is a physical confluence: in the whole
        network (before any stop rule removed branches) the parent receives
        flow from at least two segments. Selects the flow-ratio split instead of area scaling.
        False for the gage.
    theta : float
        Area-scaling exponent for the whole tree. One value per tree: the
        linear step telescopes to (A_s/A_o)**theta only for a constant
        exponent.
    pruned_ptr : ndarray[int64], shape (n+1,)
        CSR offsets into `pruned_ids`, per node.
    pruned_ids : ndarray[int64]
        Upstream neighbours of each node that are not tree members (stopped by
        a gage/waterbody, or cut by the distance cap). They get no correction
        but their flow counts in their confluence's split denominator, so a
        surviving sibling gets only its own share.
    """

    site: str
    seg_ids: np.ndarray
    parent: np.ndarray
    drainage_km2: np.ndarray
    is_junction_step: np.ndarray
    theta: float
    pruned_ptr: np.ndarray
    pruned_ids: np.ndarray

    def __post_init__(self) -> None:
        n = self.seg_ids.shape[0]
        if n < 1:
            raise ValueError("UpstreamTree needs at least the gage segment")
        for name in ("parent", "drainage_km2", "is_junction_step"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"UpstreamTree.{name} must have shape ({n},)")
        if self.pruned_ptr.shape != (n + 1,) or int(self.pruned_ptr[-1]) != self.pruned_ids.shape[0]:
            raise ValueError("UpstreamTree.pruned_ptr/pruned_ids are inconsistent")
        if self.parent[0] != -1 or (n > 1 and not np.all(self.parent[1:] < np.arange(1, n))):
            raise ValueError("UpstreamTree.parent must be -1 at the gage and precede each child")
        if n > 1 and np.any(self.parent[1:] < 0):
            raise ValueError("UpstreamTree.parent must be >= 0 for non-gage nodes")
        if not (math.isfinite(self.theta) and self.theta > 0):
            raise ValueError(f"UpstreamTree.theta must be finite and > 0, got {self.theta!r}")

    @property
    def gage_segment(self) -> int:
        """int: Segment id of the gage."""
        return int(self.seg_ids[0])

    @property
    def gage_drainage_km2(self) -> float:
        """float: A_o used by the spread."""
        return float(self.drainage_km2[0])

    @property
    def n_nodes(self) -> int:
        """int: Number of tree members, gage included."""
        return int(self.seg_ids.shape[0])

    def pruned_of(self, i: int) -> np.ndarray:
        """Non-member upstream neighbours of local node `i`."""
        return self.pruned_ids[int(self.pruned_ptr[i]) : int(self.pruned_ptr[i + 1])]


def trace_upstream_tree(
    gage_segment: int,
    rconn: Mapping[int, Sequence[int]],
    drainage_km2: Mapping[int, float],
    barrier_segments: Iterable[int],
    *,
    theta: float,
    site: str = "",
    gage_drainage_km2: Optional[float] = None,
) -> UpstreamTree:
    """Breadth-first search upstream from `gage_segment`, halting at stop segments.

    Parameters
    ----------
    gage_segment : int
        Gage segment id (tree root).
    rconn : Mapping[int, Sequence[int]]
        Reverse connectivity, `segment -> upstream segments`. Missing key or
        empty list means headwater.
    drainage_km2 : Mapping[int, float]
        Total drainage area per segment. Missing entries become NaN (callers
        drop such trees; see `build_upstream_trees`).
    barrier_segments : Iterable[int]
        Segments that end a branch (other gages, waterbodies). They are not
        members; they are recorded as pruned neighbours of the node they feed.
    theta : float
        Tree exponent, finite and > 0.
    site : str, optional
        Gage id, stored for reporting.
    gage_drainage_km2 : float, optional
        Override for A_o (e.g. a reference gage drainage area). By
        default the gage segment's modeled area is used, consistent with the
        A_s values.

    Returns
    -------
    UpstreamTree

    Raises
    ------
    ValueError
        If `gage_segment` is in `barrier_segments`, or the upstream graph is not a
        tree (a segment reached twice).
    """
    gage_segment = int(gage_segment)
    stop = frozenset(int(s) for s in barrier_segments)
    if gage_segment in stop:
        raise ValueError(f"gage segment {gage_segment} is in barrier_segments; exclude it before calling")

    seg_ids: List[int] = [gage_segment]
    parent: List[int] = [-1]
    pruned: List[List[int]] = [[]]
    seen = {gage_segment}
    queue: deque = deque([0])
    while queue:
        i = queue.popleft()
        for u in rconn.get(seg_ids[i], ()):
            u = int(u)
            if u in stop:
                pruned[i].append(u)
                continue
            if u in seen:
                raise ValueError(
                    f"segment {u} reached twice upstream of gage {gage_segment}; "
                    "reverse connectivity is not a tree"
                )
            seen.add(u)
            seg_ids.append(u)
            parent.append(i)
            pruned.append([])
            queue.append(len(seg_ids) - 1)

    n = len(seg_ids)
    seg_arr = np.asarray(seg_ids, dtype=np.int64)
    parent_arr = np.asarray(parent, dtype=np.int64)
    area = np.empty(n, dtype=np.float64)
    for i, s in enumerate(seg_ids):
        a = drainage_km2.get(s)
        area[i] = math.nan if a is None else float(a)
    if gage_drainage_km2 is not None:
        area[0] = float(gage_drainage_km2)
    junction = np.zeros(n, dtype=bool)
    for i in range(1, n):
        junction[i] = len(rconn.get(seg_ids[parent[i]], ())) >= 2
    pruned_ptr, pruned_ids = _csr(pruned)
    return UpstreamTree(
        site=str(site),
        seg_ids=seg_arr,
        parent=parent_arr,
        drainage_km2=area,
        is_junction_step=junction,
        theta=float(theta),
        pruned_ptr=pruned_ptr,
        pruned_ids=pruned_ids,
    )


def tree_distance_km(tree: UpstreamTree, length_km: Mapping[int, float]) -> np.ndarray:
    """Network distance from each node's outlet to the gage segment's outlet.

    d_0 = 0, d_i = d_p(i) + L_p(i): a node's correction sits
    at its outlet, so the water it describes still has to cross its parent,
    the parent's parent, and so on down to (and including) the gage segment.

    A parent with a missing, non-finite or non-positive length gives its whole
    subtree a distance of `inf` (fail closed: an unknown length must not keep
    far segments inside the cap).

    Returns
    -------
    ndarray[float64], shape (n,)
    """
    n = tree.n_nodes
    dist = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        p = int(tree.parent[i])
        raw = length_km.get(int(tree.seg_ids[p]))
        L = float(raw) if raw is not None else math.nan
        dist[i] = dist[p] + L if (math.isfinite(L) and L > 0) else math.inf
    return dist


def cap_tree_by_distance(
    tree: UpstreamTree, length_km: Mapping[int, float], max_reach_km: float
) -> Tuple[UpstreamTree, int]:
    """Drop nodes farther than `max_reach_km` from the gage (see `tree_distance_km`).

    A dropped node whose parent is kept is moved to that parent's pruned list,
    so it still counts in the parent's confluence denominator. Breadth-first
    order and the parent-precedes-child invariant are preserved.

    Parameters
    ----------
    tree : UpstreamTree
    length_km : Mapping[int, float]
        Segment length (km) per segment id.
    max_reach_km : float
        Finite and > 0. Nodes with distance `<= max_reach_km` are kept.

    Returns
    -------
    (UpstreamTree, int)
        The capped tree and the number of nodes dropped.
    """
    if not math.isfinite(max_reach_km) or max_reach_km <= 0:
        raise ValueError(f"max_reach_km must be finite and > 0, got {max_reach_km!r}")
    dist = tree_distance_km(tree, length_km)
    keep = dist <= max_reach_km
    n_drop = int((~keep).sum())
    if n_drop == 0:
        return tree, 0
    # Distance is non-decreasing from parent to child, so `keep` is closed under
    # ancestors and the gage (distance 0) is always kept.
    new_index = np.full(tree.n_nodes, -1, dtype=np.int64)
    kept = np.flatnonzero(keep)
    new_index[kept] = np.arange(kept.size)
    pruned: List[List[int]] = [list(tree.pruned_of(int(i))) for i in kept]
    for i in np.flatnonzero(~keep):
        p = int(tree.parent[i])
        if keep[p]:
            pruned[int(new_index[p])].append(int(tree.seg_ids[i]))
    parent = tree.parent[kept].copy()
    parent[1:] = new_index[parent[1:]]
    ptr, ids = _csr(pruned)
    return (
        replace(
            tree,
            seg_ids=tree.seg_ids[kept].copy(),
            parent=parent,
            drainage_km2=tree.drainage_km2[kept].copy(),
            is_junction_step=tree.is_junction_step[kept].copy(),
            pruned_ptr=ptr,
            pruned_ids=ids,
        ),
        n_drop,
    )
