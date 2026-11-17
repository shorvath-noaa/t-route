"""Area-scaled ("simple scaling") streamflow data assimilation: core algorithm.

The core is topology-agnostic, pure NumPy: it reads no files, configs or
t-route network objects; callers hand in plain mappings and arrays. The
HYFeatures adapter (`hyfeatures`) extracts those mappings from a built
network and is not imported here.

The method spreads a gage's analysis increment dQ_o(t) over
the gage's upstream tree:

* linear step (parent has one upstream branch in the full network):
  dQ_s = dQ_p * (A_s/A_p)**theta, which telescopes to
  dQ_o * (A_s/A_o)**theta along a chain;
* confluence step (parent has two or more upstream branches in the full
  network): dQ_b = dQ_p * Q_b / max(Q_p, sum of branch Q),
  zero when that denominator is at or below `split_floor_cms`.

Trees stop at other assimilated gages and at waterbody segments, so they are
disjoint. Segments farther than `max_reach_km` upstream are excluded.

Scope: the area-power form follows the peak-discharge scaling of Ogden &
Dawdy (2003), whose exponent 0.77 was fitted to flood-peak quantiles in a
0.17-21.2 km2 Hortonian basin. Applying it to per-timestep increments at larger
gaged basins goes beyond that study; the confluence flow split and the
distance cap are design choices of the NGWPC implementation. A scientific
review of these choices is planned.

The increment is NOT mass conserving: it is an analysis increment added to
discharge only (no depth/velocity), not routed water.

Modules
-------
tree
    `UpstreamTree`, `trace_upstream_tree`, `cap_tree_by_distance`.
forest
    `build_upstream_trees` (many gages, stop rules, drop logging).
theta
    `resolve_theta` (per-site > per-VPU > default).
spread
    `pack_trees`, `spread_increments`, `apply_increments`,
    `apply_upstream_correction`.
hyfeatures
    `build_hyfeatures_trees`: HYFeatures/HFv4.0
    network adapter (import it from `troute.scaling_da.hyfeatures`).

References
----------
Ogden, F. L., and Dawdy, D. R. (2003). Peak discharge scaling in small Hortonian
watershed. J. Hydrol. Eng. 8(2):64-73. doi:10.1061/(ASCE)1084-0699(2003)8:2(64).

The design and behavior follow the NGWPC t-route implementation of area-scaled
streamflow DA; the code here is written for OWP t-route.
"""

from troute.scaling_da.forest import TreeBuildResult, build_upstream_trees
from troute.scaling_da.tree import (
    UpstreamTree,
    trace_upstream_tree,
    cap_tree_by_distance,
    tree_distance_km,
)
from troute.scaling_da.spread import (
    DEFAULT_CHUNK_TIMESTEPS,
    DEFAULT_SPLIT_FLOOR_CMS,
    PackedTrees,
    ScalingResult,
    apply_increments,
    apply_upstream_correction,
    dq_o_matrix,
    pack_trees,
    spread_increments,
)
from troute.scaling_da.theta import DEFAULT_THETA, ThetaResolution, check_theta, resolve_theta

__all__ = [
    "DEFAULT_CHUNK_TIMESTEPS",
    "DEFAULT_SPLIT_FLOOR_CMS",
    "DEFAULT_THETA",
    "UpstreamTree",
    "PackedTrees",
    "ScalingResult",
    "ThetaResolution",
    "TreeBuildResult",
    "apply_increments",
    "apply_upstream_correction",
    "build_upstream_trees",
    "trace_upstream_tree",
    "cap_tree_by_distance",
    "check_theta",
    "dq_o_matrix",
    "pack_trees",
    "resolve_theta",
    "spread_increments",
    "tree_distance_km",
]
