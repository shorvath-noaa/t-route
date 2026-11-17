"""Run-time driver of area-scaled streamflow DA for the `main_v04` loop.

Scaling DA is streamflow nudging plus an upstream spread:

1. Routing runs with nudging as usual. At each observed gage the kernel replaces
   the routed flow with the observation (or a decaying blend after the last one)
   and records the applied correction in `nudge`; the correction travels
   downstream through routing.
2. After each loop, `ScalingDARuntime.apply` spreads each gage's recorded
   correction over its upstream tree (area-scaled along channels, split by modeled
   flow at confluences; see `troute.scaling_da.spread`) and rewrites the
   loop's published discharge there, and depth with it.
3. The corrected flow enters the routing state only on the final loop: the driver
   takes `q0` before `apply` on every other loop and
   again after it on the final one, so a following forecast starts from the
   corrected state while each loop of the analysis starts from the uncorrected one.

Sources: a crosswalked gage of a configured type is a *source* (tree root, and a
stop for every other tree) when the run's TimeSlice files hold at least one usable
observation for it and it is not held out. Held-out gages are removed from the
nudging inputs as well; gages that never report stop nothing, so the next source
downstream corrects their basin. Any other gage nudging will correct (observed, not
held out, of a type not in `site_types`) is a stop too, so the spread never adds
to a nudged flow.

Flows read by the confluence split are the loop's published flows, except at the
gage of every tree, where the nudge is removed first (published minus `nudge`),
as in the NGWPC implementation; this applies both to a tree's own gage and to a
gage that is a pruned neighbour of another tree. Gage flows are never rewritten.

The area scaling follows the peak-discharge scaling of Ogden & Dawdy (2003),
J. Hydrol. Eng. 8(2):64-73. Applying it to per-timestep increments, the confluence
flow split, the distance cap and the state semantics above are design choices of
the NGWPC implementation, ported as is; a scientific review is planned.

Approximations, stated in the configuration docs too:

* timing: an upstream segment receives the gage's correction at the same instant
  (no travel-time lag);
* the correction is an analysis increment, not routed water: mass is not
  conserved (the injected volume is logged per loop);
* depth follows the wide-channel Manning relation h ~ Q**(3/5)
  (ratio capped to [1/2, 2]); velocity is not corrected.

References: the method follows the NGWPC t-route implementation of
area-scaled streamflow DA; science reference Ogden & Dawdy (2003),
J. Hydrol. Eng. 8(2):64-73 (see `troute.scaling_da`).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from troute.scaling_da.forest import DROP_REASONS, build_upstream_trees
from troute.scaling_da.tree import UpstreamTree
from troute.scaling_da.hyfeatures import (
    ADAPTER_DROP_REASONS,
    DEFAULT_MAX_REACH_KM,
    HYFeaturesMappings,
    hyfeatures_mappings,
)
from troute.scaling_da.spread import DEFAULT_CHUNK_TIMESTEPS, PackedTrees, apply_upstream_correction, pack_trees
from troute.scaling_da.theta import ThetaResolution, check_theta, resolve_theta

__all__ = [
    "DEPTH_EXPONENT",
    "DEPTH_RATIO_CAP",
    "LoopStats",
    "ScalingDAOptions",
    "ScalingDARuntime",
    "observed_sites",
    "read_site_list",
    "read_theta_csv",
    "run_timeslice_files",
    "scaling_enabled",
]

LOG = logging.getLogger(__name__)

# Depth follows discharge as h ~ Q**DEPTH_EXPONENT (wide-channel Manning).
DEPTH_EXPONENT = 0.6
# The flow ratio used for the depth update is clipped to [1/cap, cap].
DEPTH_RATIO_CAP = 2.0

# Positions in a routing result tuple (compute_network_structured / diffusive).
_R_IDS, _R_FVD, _R_DA, _R_NUDGE = 0, 1, 3, 8
_QVD = 3  # flow, velocity, depth per timestep

# Why a gage is not a source, beyond the adapter/core drop reasons.
SOURCE_EXCLUSIONS = {
    "held_out": "listed in holdout_sites_file (not nudged, no tree, not a stop)",
    "no_observations": "no usable observation in the run's TimeSlice files (no tree, not a stop)",
}


def _chunk_steps(value: Any, default: Optional[int]) -> Optional[int]:
    """Config value -> chunk size: unset gives `default`; 0 means no chunking (None)."""
    if value is None:
        return default
    return int(value) or None


def scaling_enabled(da_config: Optional[Mapping[str, Any]]) -> bool:
    """True if `streamflow_da.streamflow_scaling` is set."""
    stream_da = dict(da_config or {}).get("streamflow_da") or {}
    return bool(stream_da.get("streamflow_scaling", False))


@dataclass(frozen=True)
class ScalingDAOptions:
    """`streamflow_scaling_parameters` as plain values (see the config model)."""

    default_theta: float = 0.77
    vpu_theta: Mapping[str, float] = field(default_factory=dict)
    per_tree_csv: Optional[Path] = None
    max_reach_km: float = DEFAULT_MAX_REACH_KM
    split_floor_cms: float = 1.0e-6
    holdout_sites_file: Optional[Path] = None
    site_types: Tuple[str, ...] = ("nwis",)
    extra_stop_segs: Tuple[int, ...] = ()
    chunk_timesteps: Optional[int] = DEFAULT_CHUNK_TIMESTEPS

    @classmethod
    def from_parameters(cls, params: Optional[Mapping[str, Any]]) -> "ScalingDAOptions":
        p = dict(params or {})
        theta = dict(p.get("theta") or {})
        return cls(
            default_theta=check_theta(theta.get("default", 0.77), "theta.default"),
            vpu_theta={str(k): check_theta(v, f"theta.by_vpu[{k!r}]") for k, v in (theta.get("by_vpu") or {}).items()},
            per_tree_csv=Path(theta["per_tree_csv"]) if theta.get("per_tree_csv") else None,
            max_reach_km=float(p.get("max_reach_km", DEFAULT_MAX_REACH_KM)),
            split_floor_cms=float(p.get("min_flow_cms", 1.0e-6)),
            holdout_sites_file=Path(p["holdout_sites_file"]) if p.get("holdout_sites_file") else None,
            site_types=tuple(str(s) for s in (p.get("site_types") or ("nwis",))),
            extra_stop_segs=tuple(int(s) for s in (p.get("extra_stop_segs") or ())),
            chunk_timesteps=_chunk_steps(p.get("spread_chunk_timesteps", None), DEFAULT_CHUNK_TIMESTEPS),
        )


def read_site_list(path: Path) -> List[str]:
    """Gage ids from a text file, one per line; blank lines and `#` comments skipped."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"streamflow_scaling_parameters.holdout_sites_file not found: {p}")
    out = []
    for raw in p.read_text().splitlines():
        s = raw.split("#", 1)[0].strip()
        if s:
            out.append(s)
    return out


def read_theta_csv(path: Path) -> Dict[str, float]:
    """`gage id -> theta` from a two-column CSV.

    A first row whose second field is not a number is a header. Ids are kept as
    strings exactly as written (leading zeros matter). Blank lines and `#`
    comments are skipped. Raises on a missing file, a malformed row, a bad value,
    a duplicate id, or a file with no rows.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"streamflow_scaling_parameters.theta.per_tree_csv not found: {p}")
    theta_of: Dict[str, float] = {}
    header_allowed = True
    for n, text in enumerate(p.read_text().splitlines()):
        where = f"{p}:{n + 1}"
        body = text.partition("#")[0].strip()
        if body == "":
            continue
        cells = list(map(str.strip, body.split(",")))
        if len(cells) < 2 or cells[0] == "":
            raise ValueError(f"{where}: expected 'gage_id,theta', got {text!r}")
        gage_id, cell = cells[0], cells[1]
        try:
            value = float(cell)
        except ValueError:
            if header_allowed:
                header_allowed = False
                continue
            raise ValueError(f"{where}: theta must be a number, got {cell!r}") from None
        header_allowed = False
        if gage_id in theta_of:
            raise ValueError(f"{where}: duplicate gage id {gage_id!r}")
        theta_of[gage_id] = check_theta(value, f"{where}: theta")
    if not theta_of:
        raise ValueError(f"streamflow_scaling_parameters.theta.per_tree_csv has no rows: {p}")
    return theta_of


def run_timeslice_files(
    da_config: Mapping[str, Any], da_sets: Sequence[Mapping[str, Any]]
) -> List[Path]:
    """Every USGS TimeSlice file any loop of the run reads (sorted, unique)."""
    if not da_config.get("usgs_timeslices_folder"):
        return []
    ts_dir = Path(da_config["usgs_timeslices_folder"])
    names = set()
    for ds in da_sets or ():
        names.update(str(f) for f in (ds or {}).get("usgs_timeslice_files", ()) or ())
    return [ts_dir / n for n in sorted(names)]


def _default_reader(path: Path):
    from troute.nhd_io import _read_timeslice_file

    return _read_timeslice_file(path)


def _observed_in_file(path: Path, qc_threshold: float, reader) -> Set[str]:
    obs, qual = reader(path)
    if obs is None or len(obs) == 0:
        return set()
    o = obs.to_numpy(dtype=np.float64)
    q = qual.reindex(index=obs.index, columns=obs.columns).to_numpy(dtype=np.float64)
    # Same screening as nhd_io.get_obs_from_timeslices: quality in [0, 1] and at or
    # above the threshold, discharge > 0.
    ok = np.isfinite(o) & (o > 0) & np.isfinite(q) & (q >= 0) & (q <= 1) & (q >= qc_threshold)
    return {str(s).strip() for s, row in zip(obs.index, ok) if row.any()}


def observed_sites(
    files: Iterable[Path],
    *,
    qc_threshold: float = 1.0,
    cpu_pool: int = 1,
    reader: Optional[Callable[[Path], Tuple[pd.DataFrame, pd.DataFrame]]] = None,
) -> Set[str]:
    """Station ids with at least one usable observation in `files`.

    `reader` returns `(observations, quality)` frames indexed by station id,
    one column per observation time, quality as a fraction (default:
    `troute.nhd_io._read_timeslice_file`).
    """
    files = list(files)
    if not files:
        return set()
    reader = reader or _default_reader
    if cpu_pool and cpu_pool > 1 and len(files) > 1:
        from joblib import Parallel, delayed

        jobs = (delayed(_observed_in_file)(f, qc_threshold, reader) for f in files)
        parts = Parallel(n_jobs=cpu_pool)(jobs)
    else:
        parts = [_observed_in_file(f, qc_threshold, reader) for f in files]
    return set().union(*parts)


@dataclass(frozen=True)
class LoopStats:
    """Diagnostics of one `ScalingDARuntime.apply` call."""

    loop: int
    final: bool
    n_trees_with_increment: int
    n_segments_changed: int
    applied_sum_cms: float
    injected_volume_m3: float
    n_depth_updates: int
    depth_ratio_median: float
    depth_ratio_p1: float
    depth_ratio_p99: float
    seconds: float


def _crosswalk(network: Any) -> Dict[int, str]:
    return {int(k): str(v) for k, v in ((getattr(network, "gages", None) or {}).get("gages", {}) or {}).items()}


def _check_unused(kind: str, keys: FrozenSet[str], unused: FrozenSet[str], known_hint: Sequence[str]) -> None:
    if not keys or not unused:
        return
    if unused == keys:
        raise ValueError(
            f"streamflow_scaling_parameters.theta.{kind}: none of its {len(keys)} key(s) match a gage in "
            f"this domain (e.g. {sorted(keys)[:5]}; the domain has e.g. {list(known_hint)[:5]}); check the "
            "key format (string ids, leading zeros)"
        )
    LOG.warning(
        "scaling DA: theta.%s has %d key(s) matching no gage in this domain (no effect here): %s%s",
        kind, len(unused), sorted(unused)[:10], " ..." if len(unused) > 10 else "",
    )


class ScalingDARuntime:
    """Static setup (trees, sources, holdout) and the per-loop upstream spread.

    Build with `build`; it returns None when scaling is off. Attributes
    `sources`, `held_out`, `no_observations` (site id sets), `trees`,
    `report` (one row per gage of a configured type) and `stats` (one
    `LoopStats` per `apply` call) are for logging and tests.
    """

    def __init__(
        self,
        *,
        trees: Mapping[str, UpstreamTree],
        options: ScalingDAOptions,
        held_out_segs: Iterable[int] = (),
        sources: Iterable[str] = (),
        held_out: Iterable[str] = (),
        no_observations: Iterable[str] = (),
        report: Optional[pd.DataFrame] = None,
    ):
        self.options = options
        self.trees: Dict[str, UpstreamTree] = dict(trees)
        self.sources: FrozenSet[str] = frozenset(sources) if sources else frozenset(self.trees)
        self.held_out: FrozenSet[str] = frozenset(held_out)
        self.no_observations: FrozenSet[str] = frozenset(no_observations)
        self.held_out_segs: FrozenSet[int] = frozenset(int(s) for s in held_out_segs)
        self.report = report if report is not None else pd.DataFrame()
        self.stats: List[LoopStats] = []
        self._loop = 0

        segs = set()
        for t in self.trees.values():
            segs.update(int(s) for s in t.seg_ids)
            segs.update(int(s) for s in t.pruned_ids)
        # Every segment whose flow the spread reads (tree members and pruned neighbours).
        self.needed_segs = np.asarray(sorted(segs), dtype=np.int64)
        compact = {int(s): i for i, s in enumerate(self.needed_segs)}
        self.packed: PackedTrees
        self.packed, dropped = pack_trees(self.trees, compact)
        if dropped:  # cannot happen: the compact index holds every member and neighbour
            raise RuntimeError(f"scaling DA: trees could not be packed: {dropped}")
        self._tree_of_gage = {int(t.gage_segment): i for i, t in enumerate(self.trees[s] for s in self.packed.sites)}

    # ------------------------------------------------------------------ setup
    @classmethod
    def build(
        cls,
        network: Any,
        da_config: Mapping[str, Any],
        da_sets: Sequence[Mapping[str, Any]] = (),
        *,
        cpu_pool: int = 1,
        observed: Optional[Iterable[str]] = None,
    ) -> Optional["ScalingDARuntime"]:
        """Set up scaling DA for a built network, or return None if it is off.

        Parameters
        ----------
        network : HYFeaturesNetwork
            Built network (HYFeatures only; NHD networks raise NotImplementedError).
        da_config : Mapping
            `compute_parameters.da_config` as a dict.
        da_sets : Sequence[Mapping]
            The run's TimeSlice file sets (`build_da_sets`), read for the source roster.
        cpu_pool : int
            Workers for reading the TimeSlice roster.
        observed : Iterable[str], optional
            Station ids with observations; skips reading TimeSlice files (tests).
        """
        if not scaling_enabled(da_config):
            return None
        t0_setup = time.perf_counter()
        stream_da = da_config.get("streamflow_da") or {}
        if not stream_da.get("streamflow_nudging", False):
            raise ValueError("streamflow_da.streamflow_scaling requires streamflow_da.streamflow_nudging: True")
        if not hasattr(network, "segment_attributes"):
            raise NotImplementedError("area-scaled streamflow DA supports HYFeatures networks only")
        opts = ScalingDAOptions.from_parameters(stream_da.get("streamflow_scaling_parameters"))

        m: HYFeaturesMappings = hyfeatures_mappings(
            network, site_types=opts.site_types, extra_stop_segs=opts.extra_stop_segs
        )
        crosswalk = _crosswalk(network)
        seg_of_site: Dict[str, int] = {}
        for seg, site in crosswalk.items():
            seg_of_site.setdefault(site, seg)
        domain_sites = set(seg_of_site) | set(m.gage_report["site"].astype(str))

        held_out: Set[str] = set()
        if opts.holdout_sites_file is not None:
            listed = read_site_list(opts.holdout_sites_file)
            unknown = sorted(set(listed) - domain_sites)
            if unknown:
                raise ValueError(
                    f"holdout_sites_file (streamflow_scaling_parameters): {len(unknown)} id(s) are not "
                    f"gages of this domain and would withhold nothing, e.g. {unknown[:5]}"
                )
            held_out = set(listed)
            not_assim = sorted(held_out - set(m.gage_segments))
            if not_assim:
                LOG.warning("scaling DA: %d held-out gage(s) are not assimilated in this domain anyway: %s",
                            len(not_assim), not_assim[:10])
        held_out_segs = {seg_of_site[s] for s in held_out if s in seg_of_site}

        if observed is None:
            files = run_timeslice_files(da_config, da_sets)
            qc = da_config.get("qc_threshold")
            observed = observed_sites(
                files,
                qc_threshold=1.0 if qc is None else float(qc),
                cpu_pool=cpu_pool or 1,
            )
            if not observed:
                raise ValueError(
                    "scaling DA: the run's USGS TimeSlice files hold no usable observations "
                    f"({len(files)} file(s) in {da_config.get('usgs_timeslices_folder')!r}); "
                    "check usgs_timeslices_folder and the simulation period"
                )
        observed = {str(s).strip() for s in observed}

        candidates = list(m.gage_segments)  # assimilable sites of the configured types, ordered
        no_obs = {s for s in candidates if s not in observed and s not in held_out}
        sources = [s for s in candidates if s in observed and s not in held_out]
        if candidates and not sources:
            LOG.warning("scaling DA: none of the %d gage(s) in this domain has observations for this run; "
                        "the upstream spread will do nothing", len(candidates))

        per_site = read_theta_csv(opts.per_tree_csv) if opts.per_tree_csv is not None else {}
        theta: ThetaResolution = resolve_theta(
            sources, default=opts.default_theta, by_vpu=opts.vpu_theta, per_site=per_site,
            site_vpu=m.site_vpu,
        )
        _check_unused("per_tree_csv", frozenset(per_site), frozenset(per_site) - frozenset(domain_sites),
                      sorted(domain_sites))
        domain_vpus = set(m.site_vpu.values())
        _check_unused("by_vpu", frozenset(opts.vpu_theta), frozenset(opts.vpu_theta) - frozenset(domain_vpus),
                      sorted(domain_vpus))

        # Every gage nudging will correct is a stop, whatever its type, so the spread
        # never adds to a nudged flow.
        source_segs = {m.gage_segments[s] for s in sources}
        nudged_other = {seg for seg, site in crosswalk.items()
                        if site in observed and site not in held_out and seg not in source_segs}
        if nudged_other:
            LOG.info("scaling DA: %d other nudged gage segment(s) (types not in site_types) stop the trees",
                     len(nudged_other))

        res = build_upstream_trees(
            m.rconn,
            {s: m.gage_segments[s] for s in sources},
            m.drainage_km2,
            site_theta=theta.site_theta,
            default_theta=opts.default_theta,
            lake_segments=m.lake_segments,
            extra_stop_segs=m.extra_stop_segs | frozenset(nudged_other),
            stop_at_lakes=True,
            length_km=m.length_km,
            max_reach_km=opts.max_reach_km,
        )

        report = m.gage_report.copy()
        status = []
        for s in report["site"].astype(str):
            if s in m.dropped:
                status.append(m.dropped[s])
            elif s in held_out:
                status.append("held_out")
            elif s in no_obs:
                status.append("no_observations")
            elif s in res.dropped:
                status.append(res.dropped[s])
            else:
                status.append("source")
        report["status"] = status
        report["n_tree_segments"] = [res.trees[s].n_nodes if s in res.trees else 0 for s in report["site"]]
        report["theta"] = [theta.site_theta.get(s, np.nan) for s in report["site"]]
        report["theta_source"] = [theta.source_by_site.get(s, "") for s in report["site"]]

        rt = cls(
            trees=res.trees, options=opts, held_out_segs=held_out_segs, sources=sources,
            held_out=held_out, no_observations=no_obs, report=report,
        )
        n_src = {k: v for k, v in pd.Series(theta.source_by_site).value_counts().items()} if sources else {}
        LOG.info(
            "scaling DA: %d source gage(s), %d tree(s) covering %d segment(s); %d held out, %d without "
            "observations; theta from csv/vpu/default: %d/%d/%d; setup %.2f s",
            len(sources), len(res.trees), sum(t.n_nodes for t in res.trees.values()), len(held_out),
            len(no_obs), n_src.get("site", 0), n_src.get("vpu", 0), n_src.get("default", 0),
            time.perf_counter() - t0_setup,
        )
        if no_obs:
            LOG.info("scaling DA: gage(s) without observations (not stops): %s%s", sorted(no_obs)[:10],
                     " ..." if len(no_obs) > 10 else "")
        return rt

    # ------------------------------------------------------------ nudging inputs
    def nudging_inputs(self, usgs_df: pd.DataFrame, lastobs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """`usgs_df` and `lastobs_df` without held-out gages (index = segment id)."""
        if not self.held_out_segs:
            return usgs_df, lastobs_df

        def drop(df):
            if df is None or df.empty:
                return df
            keep = ~df.index.isin(list(self.held_out_segs))
            return df if keep.all() else df.loc[keep]

        return drop(usgs_df), drop(lastobs_df)

    # ------------------------------------------------------------------ per loop
    def _locate(self, run_results: Sequence[Sequence[Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """`(result index, local row)` of every needed segment."""
        ids = [np.asarray(r[_R_IDS], dtype=np.int64).ravel() for r in run_results]
        lens = np.asarray([a.size for a in ids], dtype=np.int64)
        all_ids = np.concatenate(ids) if ids else np.empty(0, dtype=np.int64)
        res_of = np.repeat(np.arange(len(ids), dtype=np.int64), lens)
        local = np.concatenate([np.arange(n, dtype=np.int64) for n in lens]) if ids else np.empty(0, dtype=np.int64)
        order = np.argsort(all_ids, kind="stable")
        sids = all_ids[order]
        need = self.needed_segs
        pos = np.searchsorted(sids, need)
        posc = np.minimum(pos, max(sids.size - 1, 0))
        found = (pos < sids.size) & (sids[posc] == need) if sids.size else np.zeros(need.size, dtype=bool)
        if not found.all():
            miss = need[~found]
            raise RuntimeError(
                f"scaling DA: {miss.size} tree segment(s) have no row in the routing results "
                f"(e.g. {miss[:5].tolist()})"
            )
        nxt = np.minimum(pos + 1, sids.size - 1)
        dup = (pos + 1 < sids.size) & (sids[nxt] == need)
        if dup.any():
            raise RuntimeError(f"scaling DA: segment(s) appear in more than one result: {need[dup][:5].tolist()}")
        hit = order[pos]
        return res_of[hit], local[hit]

    def apply(self, run_results: Sequence[Sequence[Any]], *, dt: float, final: bool) -> bool:
        """Spread this loop's gage corrections upstream; rewrite `run_results` in place.

        Returns True when `final` and any flow changed, i.e. when the caller must
        take `q0` again so the corrected state reaches the next run.
        """
        t0_setup = time.perf_counter()
        loop = self._loop
        self._loop += 1
        packed = self.packed
        if packed.n_trees == 0 or not run_results:
            return False

        res_idx, local = self._locate(run_results)
        groups = []  # (result, compact rows, local rows)
        for k in np.unique(res_idx):
            sel = np.flatnonzero(res_idx == k)
            groups.append((int(k), sel, local[sel]))

        nts = None
        for k, _, _ in groups:
            w = np.asarray(run_results[k][_R_FVD]).shape[1]
            if w % _QVD:
                raise RuntimeError(f"scaling DA: result {k} has {w} columns, not a multiple of {_QVD}")
            if nts is None:
                nts = w // _QVD
            elif w // _QVD != nts:
                raise RuntimeError("scaling DA: results disagree on the number of timesteps")
        q = np.empty((self.needed_segs.size, nts), dtype=np.float32)
        for k, rows_c, rows_l in groups:
            q[rows_c] = np.asarray(run_results[k][_R_FVD])[rows_l, 0::_QVD]

        # Gage corrections: nudge column 0 is the initial condition (never written by
        # the kernel); columns 1..nts align with the published timesteps.
        dq_o = np.zeros((packed.n_trees, nts), dtype=np.float64)
        for r in run_results:
            gids = np.asarray(r[_R_DA][0]).astype(np.int64).ravel() if len(r[_R_DA]) else np.empty(0, np.int64)
            if gids.size == 0:
                continue
            nud = np.asarray(r[_R_NUDGE])
            if nud.shape != (gids.size, nts + 1):
                raise RuntimeError(f"scaling DA: nudge array has shape {nud.shape}, expected ({gids.size}, {nts + 1})")
            for i, seg in enumerate(gids.tolist()):
                ti = self._tree_of_gage.get(int(seg))
                if ti is not None:
                    dq_o[ti] = nud[i, 1:]
        # Background (pre-nudge) flow at every tree's gage, as NGWPC does: the
        # confluence split, including a gage that is another tree's pruned
        # neighbour, reads modeled flow there. Gage rows are never written back.
        root_rows = packed.rows[packed.roots]
        root_flow = q[root_rows].astype(np.float64) - dq_o
        q[root_rows] = root_flow.astype(q.dtype)

        old = q[packed.rows].copy()
        res = apply_upstream_correction(
            q, packed, dq_o,
            split_floor_cms=self.options.split_floor_cms,
            root_flow=root_flow,
            include_root=False,
            chunk_timesteps=self.options.chunk_timesteps,
            clip_min=0.0,
            dt_s=float(dt),
        )

        changed = np.flatnonzero(res.changed)
        ratios = []
        if changed.size:
            rows_changed = packed.rows[changed]  # compact rows
            is_changed = np.zeros(self.needed_segs.size, dtype=bool)
            is_changed[rows_changed] = True
            old_of = np.zeros((self.needed_segs.size, nts), dtype=np.float32)
            old_of[rows_changed] = old[changed]
            cap = DEPTH_RATIO_CAP
            for k, rows_c, rows_l in groups:
                m = is_changed[rows_c]
                if not m.any():
                    continue
                rc, rl = rows_c[m], rows_l[m]
                fvd = run_results[k][_R_FVD]
                q_new = q[rc]
                q_old = old_of[rc].astype(np.float64)
                depth = fvd[rl, 2::_QVD].astype(np.float64)
                live = (q_old > self.options.split_floor_cms) & (q_new > self.options.split_floor_cms) & (depth > 0.0)
                if live.any():
                    ratio = np.where(live, q_new / np.where(live, q_old, 1.0), 1.0)
                    np.clip(ratio, 1.0 / cap, cap, out=ratio)
                    factor = np.where(live, ratio ** DEPTH_EXPONENT, 1.0)
                    fvd[rl, 2::_QVD] = (depth * factor).astype(fvd.dtype)
                    ratios.append(factor[live & (q_new != q_old)])
                fvd[rl, 0::_QVD] = q_new.astype(fvd.dtype)

        dh = np.concatenate(ratios) if ratios else np.empty(0)
        stats = LoopStats(
            loop=loop,
            final=bool(final),
            n_trees_with_increment=int(np.count_nonzero(np.any(dq_o != 0.0, axis=1))),
            n_segments_changed=res.n_segments_changed,
            applied_sum_cms=float(res.applied_sum_cms),
            injected_volume_m3=float(res.injected_volume_m3 or 0.0),
            n_depth_updates=int(dh.size),
            depth_ratio_median=float(np.median(dh)) if dh.size else math.nan,
            depth_ratio_p1=float(np.percentile(dh, 1)) if dh.size else math.nan,
            depth_ratio_p99=float(np.percentile(dh, 99)) if dh.size else math.nan,
            seconds=time.perf_counter() - t0_setup,
        )
        self.stats.append(stats)
        LOG.info(
            "scaling DA loop %d%s: %d tree(s) with a gage correction, %d segment(s) changed, "
            "injected volume %.4g m3 (not mass conserving); depth ratio median %.4f [p1 %.4f, p99 %.4f] "
            "over %d update(s); %.2f s",
            loop, " (final: correction enters the hand-off state)" if final else "",
            stats.n_trees_with_increment, stats.n_segments_changed, stats.injected_volume_m3,
            stats.depth_ratio_median, stats.depth_ratio_p1, stats.depth_ratio_p99, stats.n_depth_updates,
            stats.seconds,
        )
        return bool(final and stats.n_segments_changed > 0)
