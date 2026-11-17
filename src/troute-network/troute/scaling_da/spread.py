"""Spread gage analysis increments over their upstream trees (NumPy reference).

Layout follows t-route: flows are segment-major, `q[row, t]`, where `row`
is a segment's row in the routed-results array and `t` a routing timestep.

Per tree and timestep, a multiplier M is built from the gage up:

    M_o = 1                                          at the gage
    M_s = M_p * clip((A_s/A_p)**theta, 0, 1)         linear step
    M_s = M_p * clip(Q_s / max(Q_p, sum_b Q_b), 0, 1)
                                                     confluence step, if max(...) > split_floor_cms
    M_s = 0                                          confluence step, otherwise

and the increment is dQ_s(t) = dQ_o(t) * M_s(t). The branch sum
sum_b Q_b runs over all of the parent's upstream branches in the
network: tree members and pruned ones (stopped or capped), which receive
nothing. With the clamps, |M_s| <= 1: a correction never grows away
from the gage, and confluence shares sum to at most 1.

Computation is float64 regardless of the input dtype. Linear factors are static
and hoisted; the multiplier is propagated one BFS depth level at a time across
all trees at once, so Python-level work scales with tree depth, not tree size.

Deviation from NGWPC's kernel (on purpose): a NaN confluence share (NaN branch
flow) gives 0, not NaN; non-finite innovations are treated as 0 and logged.

References: the method follows the NGWPC t-route implementation of
area-scaled streamflow DA; science reference Ogden & Dawdy (2003),
J. Hydrol. Eng. 8(2):64-73 (see `troute.scaling_da`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np

from troute.scaling_da.tree import UpstreamTree

__all__ = [
    "DEFAULT_CHUNK_TIMESTEPS",
    "DEFAULT_SPLIT_FLOOR_CMS",
    "PackedTrees",
    "ScalingResult",
    "apply_increments",
    "apply_upstream_correction",
    "dq_o_matrix",
    "pack_trees",
    "spread_increments",
]

LOG = logging.getLogger(__name__)

# Divisions by zero and 0/0 are handled explicitly after the division.
_IGNORE_FP = {"divide": "ignore", "invalid": "ignore"}

# Confluence denominator floor (m³/s); NGWPC default.
DEFAULT_SPLIT_FLOOR_CMS = 1e-6


@dataclass(frozen=True)
class PackedTrees:
    """All trees flattened into global arrays, mapped to result rows.

    Node `k` belongs to tree `tree_of[k]`; tree `i` owns nodes
    `tree_ptr[i]:tree_ptr[i+1]` with its root first.

    Attributes
    ----------
    sites : tuple[str, ...]
        Site per tree; row order of the `dq_o` input.
    tree_ptr : ndarray[int64], (n_trees+1,)
    rows : ndarray[int64], (N,)
        Result-array row of each node. Unique across all trees.
    parent : ndarray[int64], (N,)
        Global node index of the parent; -1 for roots.
    tree_of : ndarray[int64], (N,)
    is_junction_step : ndarray[bool], (N,)
    lin_factor : ndarray[float64], (N,)
        Clamped (A_s/A_p)**theta for linear steps; 1 for roots; unused
        for confluence steps.
    levels : tuple[ndarray[int64], ...]
        Node indices at BFS depth 1, 2, ... (depth 0 = roots).
    pruned_ptr : ndarray[int64], (N+1,)
    pruned_rows : ndarray[int64]
        Result rows of each node's pruned neighbours (CSR over nodes).
    n_area_inversions : int
        Linear steps whose area grows upstream (factor clamped to 1). Indicates
        hydrofabric area inconsistencies; reported, not fatal.
    """

    sites: Tuple[str, ...]
    tree_ptr: np.ndarray
    rows: np.ndarray
    parent: np.ndarray
    tree_of: np.ndarray
    is_junction_step: np.ndarray
    lin_factor: np.ndarray
    levels: Tuple[np.ndarray, ...]
    pruned_ptr: np.ndarray
    pruned_rows: np.ndarray
    n_area_inversions: int

    @property
    def n_trees(self) -> int:
        return len(self.sites)

    @property
    def n_nodes(self) -> int:
        return int(self.rows.shape[0])

    @property
    def roots(self) -> np.ndarray:
        """Global node index of each tree's root (gage)."""
        return self.tree_ptr[:-1]


def _linear_factors(tree: UpstreamTree) -> Tuple[np.ndarray, int]:
    n = tree.n_nodes
    f = np.ones(n, dtype=np.float64)
    if n == 1:
        return f, 0
    a = tree.drainage_km2
    p = tree.parent[1:]
    ap = a[p]
    with np.errstate(**_IGNORE_FP):
        raw = np.where(ap > 0, (a[1:] / ap) ** tree.theta, 0.0)
    lin = ~tree.is_junction_step[1:]
    inversions = int(np.count_nonzero(lin & (raw > 1.0)))
    raw = np.where(np.isfinite(raw), raw, 0.0)
    f[1:] = np.clip(raw, 0.0, 1.0)
    return f, inversions


def pack_trees(
    trees: Mapping[str, UpstreamTree], seg_to_row: Mapping[int, int]
) -> Tuple[PackedTrees, Dict[str, str]]:
    """Flatten trees and map segment ids to result-array rows.

    A tree is dropped (whole) if any member or pruned neighbour has no row:
    a missing member can't be corrected, and a missing pruned neighbour would
    shrink a confluence denominator and over-credit the surviving sibling.

    Returns
    -------
    (PackedTrees, dict[str, str])
        Packed trees and `site -> reason` for dropped trees.

    Raises
    ------
    ValueError
        If two trees map to the same row (trees must be disjoint).
    """
    dropped: Dict[str, str] = {}
    sites: List[str] = []
    rows_l, parent_l, junc_l, lin_l, depth_l, pptr_l, prow_l = [], [], [], [], [], [], []
    ptr = [0]
    inversions = 0
    for gage_id, tree in trees.items():
        try:
            rows = np.fromiter((seg_to_row[int(s)] for s in tree.seg_ids), dtype=np.int64, count=tree.n_nodes)
            prow = np.fromiter((seg_to_row[int(s)] for s in tree.pruned_ids), dtype=np.int64, count=tree.pruned_ids.size)
        except KeyError as e:
            dropped[str(gage_id)] = f"segment {e.args[0]} has no result row"
            continue
        lin, inv = _linear_factors(tree)
        inversions += inv
        depth = np.zeros(tree.n_nodes, dtype=np.int64)
        for i in range(1, tree.n_nodes):
            depth[i] = depth[tree.parent[i]] + 1
        off = ptr[-1]
        par = tree.parent.copy()
        par[1:] += off
        sites.append(str(gage_id))
        rows_l.append(rows)
        parent_l.append(par)
        junc_l.append(tree.is_junction_step)
        lin_l.append(lin)
        depth_l.append(depth)
        pptr_l.append(np.diff(tree.pruned_ptr))
        prow_l.append(prow)
        ptr.append(off + tree.n_nodes)

    def cat(parts, dtype):
        return np.concatenate(parts).astype(dtype, copy=False) if parts else np.empty(0, dtype=dtype)

    rows = cat(rows_l, np.int64)
    if np.unique(rows).size != rows.size:
        raise ValueError("pack_trees: trees share result rows; trees must be disjoint")
    tree_ptr = np.asarray(ptr, dtype=np.int64)
    depth = cat(depth_l, np.int64)
    order = np.argsort(depth, kind="stable")
    bounds = np.searchsorted(depth[order], np.arange(1, int(depth.max(initial=0)) + 2))
    levels = tuple(order[bounds[d - 1] : bounds[d]] for d in range(1, len(bounds)))
    pruned_ptr = np.zeros(rows.size + 1, dtype=np.int64)
    np.cumsum(cat(pptr_l, np.int64), out=pruned_ptr[1:])
    packed = PackedTrees(
        sites=tuple(sites),
        tree_ptr=tree_ptr,
        rows=rows,
        parent=cat(parent_l, np.int64),
        tree_of=np.repeat(np.arange(len(sites), dtype=np.int64), np.diff(tree_ptr)),
        is_junction_step=cat(junc_l, bool),
        lin_factor=cat(lin_l, np.float64),
        levels=levels,
        pruned_ptr=pruned_ptr,
        pruned_rows=cat(prow_l, np.int64),
        n_area_inversions=inversions,
    )
    if dropped:
        LOG.warning("scaling DA: dropped %d tree(s) with segments missing from results: %s",
                    len(dropped), list(dropped.items())[:5])
    if inversions:
        LOG.warning("scaling DA: %d linear step(s) have drainage area increasing upstream; "
                    "their area factor is clamped to 1", inversions)
    return packed, dropped


def dq_o_matrix(packed: PackedTrees, dq_by_site: Mapping[str, np.ndarray], nt: int) -> np.ndarray:
    """`(n_trees, nt)` float64 innovations in `packed.sites` order; zeros for absent sites."""
    out = np.zeros((packed.n_trees, nt), dtype=np.float64)
    for i, s in enumerate(packed.sites):
        v = dq_by_site.get(s)
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float64).ravel()
        if v.shape[0] != nt:
            raise ValueError(f"dq for site {s} has {v.shape[0]} steps, expected {nt}")
        out[i] = v
    return out


def _group_sum(values: np.ndarray, owners: np.ndarray, n_owner: int) -> np.ndarray:
    """Sum rows of `values` (k, c) into `n_owner` rows by `owners` (k,)."""
    out = np.zeros((n_owner, values.shape[1]), dtype=np.float64)
    if owners.size:
        order = np.argsort(owners, kind="stable")
        o = owners[order]
        starts = np.flatnonzero(np.r_[True, o[1:] != o[:-1]])
        out[o[starts]] = np.add.reduceat(values[order], starts, axis=0)
    return out


def _validated_inputs(packed, q, dq_o, root_flow):
    q = np.asarray(q)
    if q.ndim != 2:
        raise ValueError("q must be 2-D (rows, timesteps)")
    nt = q.shape[1]
    dq_o = np.asarray(dq_o, dtype=np.float64)
    if dq_o.shape != (packed.n_trees, nt):
        raise ValueError(f"dq_o shape {dq_o.shape} != ({packed.n_trees}, {nt})")
    bad = ~np.isfinite(dq_o)
    if bad.any():
        LOG.warning("scaling DA: %d non-finite gage increment value(s) treated as 0", int(bad.sum()))
        dq_o = np.where(bad, 0.0, dq_o)
    if root_flow is not None:
        root_flow = np.asarray(root_flow, dtype=np.float64)
        if root_flow.shape != (packed.n_trees, nt):
            raise ValueError(f"root_flow shape {root_flow.shape} != ({packed.n_trees}, {nt})")
    if packed.n_nodes and int(packed.rows.max()) >= q.shape[0]:
        raise ValueError("packed rows exceed q's row count")
    return q, dq_o, root_flow


def _iter_chunks(
    packed: PackedTrees,
    q: np.ndarray,
    dq_o: np.ndarray,
    root_flow: Optional[np.ndarray],
    split_floor_cms: float,
    include_root: bool,
    chunk_timesteps: Optional[int],
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Yield `(t0, t1, inc)` with `inc` of shape `(n_nodes, t1 - t0)`.

    Each chunk reads only `q[:, t0:t1]`, so a caller may write a chunk back
    into `q` before the next one is computed.
    """
    N, nt = packed.n_nodes, q.shape[1]
    if N == 0 or nt == 0:
        return
    step = nt if not chunk_timesteps else max(1, int(chunk_timesteps))
    junc = np.flatnonzero(packed.is_junction_step)
    jpar = packed.parent[junc]
    powner = np.repeat(np.arange(N, dtype=np.int64), np.diff(packed.pruned_ptr))
    roots = packed.roots
    for t0 in range(0, nt, step):
        t1 = min(nt, t0 + step)
        qn = q[packed.rows, t0:t1].astype(np.float64)
        if root_flow is not None:
            qn[roots] = root_flow[:, t0:t1]
        mult = np.repeat(packed.lin_factor[:, None], t1 - t0, axis=1)
        if junc.size:
            bsum = _group_sum(qn[junc], jpar, N)
            if powner.size:
                bsum += _group_sum(q[packed.pruned_rows, t0:t1].astype(np.float64), powner, N)
            denom = np.maximum(qn[jpar], bsum[jpar])  # NaN-propagating
            with np.errstate(**_IGNORE_FP):
                share = np.where(denom > split_floor_cms, qn[junc] / denom, 0.0)
            share = np.clip(np.where(np.isfinite(share), share, 0.0), 0.0, 1.0)
            mult[junc] = share
        for lvl in packed.levels:
            mult[lvl] *= mult[packed.parent[lvl]]
        mult *= dq_o[packed.tree_of, t0:t1]
        if not include_root:
            mult[roots] = 0.0
        yield t0, t1, mult


def spread_increments(
    packed: PackedTrees,
    q: np.ndarray,
    dq_o: np.ndarray,
    *,
    split_floor_cms: float = DEFAULT_SPLIT_FLOOR_CMS,
    root_flow: Optional[np.ndarray] = None,
    include_root: bool = False,
    chunk_timesteps: Optional[int] = None,
) -> np.ndarray:
    """Per-node increments dQ_s(t), shape `(n_nodes, nt)`, float64.

    Materializes the full increment array; for production write-back use
    `apply_upstream_correction`, which never holds more than one time chunk.

    Parameters
    ----------
    packed : PackedTrees
    q : ndarray, (n_rows, nt)
        Modeled flow used for the confluence split, any float dtype. Upstream of
        a gage, nudging does not change flow, so the published flows are the
        background there.
    dq_o : ndarray, (n_trees, nt)
        Gage increment per tree and timestep (e.g. the kernel's `nudge`).
        Non-finite values are treated as 0 (logged).
    split_floor_cms : float, optional
        Confluence denominator floor; at or below it the branch gets nothing.
    root_flow : ndarray, (n_trees, nt), optional
        Flow at each gage to use in place of `q[root rows]`: pass the
        pre-nudge background when `q` holds the nudged (published) value.
    include_root : bool, optional
        If False (default) roots get 0: the gage itself is corrected by nudging.
    chunk_timesteps : int, optional
        Process this many timesteps at a time (result is identical).
    """
    q, dq_o, root_flow = _validated_inputs(packed, q, dq_o, root_flow)
    out = np.zeros((packed.n_nodes, q.shape[1]), dtype=np.float64)
    for t0, t1, inc in _iter_chunks(packed, q, dq_o, root_flow, split_floor_cms, include_root, chunk_timesteps):
        out[:, t0:t1] = inc
    return out


def _apply_block(
    q: np.ndarray, rows: np.ndarray, t0: int, t1: int, inc: np.ndarray, clip_min: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Write `inc` into `q[rows, t0:t1]`; return `(live node indices, applied block)`."""
    live = np.flatnonzero(np.any(inc != 0.0, axis=1))
    if live.size == 0:
        return live, np.empty((0, t1 - t0), dtype=np.float64)
    r = rows[live]
    old = q[r, t0:t1].astype(np.float64)
    new = old + inc[live]
    if clip_min is not None:
        np.maximum(new, clip_min, out=new)
    stored = new.astype(q.dtype)
    q[r, t0:t1] = stored
    return live, stored.astype(np.float64) - old


def apply_increments(
    q: np.ndarray, rows: np.ndarray, inc: np.ndarray, *, clip_min: Optional[float] = 0.0
) -> np.ndarray:
    """Add `inc` to `q[rows]` in place; return the increments actually applied.

    Rows whose increments are all zero are not written (bit-identical). The
    sum is formed in float64, floored at `clip_min` (None: no floor), and cast
    to `q.dtype`. The returned array is `new - old` in float64 of the stored
    values, so it reflects both the floor and the float32 rounding.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if inc.shape != (rows.size, q.shape[1]):
        raise ValueError(f"inc shape {inc.shape} != ({rows.size}, {q.shape[1]})")
    applied = np.zeros(inc.shape, dtype=np.float64)
    live, block = _apply_block(q, rows, 0, q.shape[1], inc, clip_min)
    applied[live] = block
    return applied


@dataclass(frozen=True)
class ScalingResult:
    """Diagnostics from `apply_upstream_correction`.

    Attributes
    ----------
    rows : ndarray[int64]
        Result rows of the tree nodes (`packed.rows`).
    changed : ndarray[bool], (n_nodes,)
        Nodes whose stored flow changed in at least one timestep.
    applied_sum_cms : float
        Sum of applied increments over nodes and timesteps (m³/s).
    injected_volume_m3 : float or None
        `applied_sum_cms * dt_s` when `dt_s` was given. The method is not
        mass conserving; this is the size of the non-conservation.
    applied : ndarray[float64], (n_nodes, nt), or None
        Increments actually written (after flooring and dtype cast); only with
        `return_applied=True` (it is as large as the trees times `nt`).
    """

    rows: np.ndarray
    changed: np.ndarray
    applied_sum_cms: float
    injected_volume_m3: Optional[float]
    applied: Optional[np.ndarray] = None

    @property
    def n_segments_changed(self) -> int:
        return int(np.count_nonzero(self.changed))


# Default time chunk for `apply_upstream_correction` (bounds temporaries to a few
# arrays of n_nodes x 24; results do not depend on it).
DEFAULT_CHUNK_TIMESTEPS = 24


def apply_upstream_correction(
    q: np.ndarray,
    packed: PackedTrees,
    dq_o: np.ndarray,
    *,
    split_floor_cms: float = DEFAULT_SPLIT_FLOOR_CMS,
    root_flow: Optional[np.ndarray] = None,
    include_root: bool = False,
    chunk_timesteps: Optional[int] = DEFAULT_CHUNK_TIMESTEPS,
    clip_min: Optional[float] = 0.0,
    dt_s: Optional[float] = None,
    return_applied: bool = False,
) -> ScalingResult:
    """Spread `dq_o` over the trees and add it to `q` in place, chunk by chunk.

    Same numbers as `spread_increments` followed by
    `apply_increments`, but only one time chunk of temporaries is alive
    at a time. Segments outside the trees are never written. See
    `spread_increments` for the parameters.
    """
    q, dq_o, root_flow = _validated_inputs(packed, q, dq_o, root_flow)
    rows = packed.rows
    changed = np.zeros(packed.n_nodes, dtype=bool)
    applied = np.zeros((packed.n_nodes, q.shape[1]), dtype=np.float64) if return_applied else None
    total = 0.0
    for t0, t1, inc in _iter_chunks(packed, q, dq_o, root_flow, split_floor_cms, include_root, chunk_timesteps):
        live, block = _apply_block(q, rows, t0, t1, inc, clip_min)
        if live.size:
            changed[live] |= np.any(block != 0.0, axis=1)
            total += float(block.sum())
            if applied is not None:
                applied[live, t0:t1] = block
    return ScalingResult(
        rows=rows,
        changed=changed,
        applied_sum_cms=total,
        injected_volume_m3=total * dt_s if dt_s is not None else None,
        applied=applied,
    )
