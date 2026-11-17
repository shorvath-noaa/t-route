"""Resolve the per-tree area-scaling exponent theta.

Precedence, most specific first: per-site value, then per-VPU value (by the
gage's VPU), then the default. No file reading here: the caller (config layer)
parses CSVs/tables and passes plain mappings.

References: the method follows the NGWPC t-route implementation of
area-scaled streamflow DA; science reference Ogden & Dawdy (2003),
J. Hydrol. Eng. 8(2):64-73 (see `troute.scaling_da`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Mapping, Optional

__all__ = ["DEFAULT_THETA", "ThetaResolution", "resolve_theta", "check_theta"]

# Default exponent (as in the NGWPC implementation): the flood-peak quantile
# scaling exponent of Ogden & Dawdy (2003), fitted in a 0.17-21.2 km2 Hortonian
# basin; not validated for larger basins or for per-timestep increments.
DEFAULT_THETA = 0.77


def check_theta(value: float, label: str = "theta") -> float:
    """Return `float(value)` if finite and > 0, else raise `ValueError`."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number, got {value!r}") from None
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    return v


@dataclass(frozen=True)
class ThetaResolution:
    """Result of `resolve_theta`.

    Attributes
    ----------
    site_theta : dict[str, float]
        Exponent for every requested site.
    source_by_site : dict[str, str]
        `"site"`, `"vpu"` or `"default"` per site.
    unused_site_keys : frozenset[str]
        `per_site` keys matching no requested site.
    unused_vpu_keys : frozenset[str]
        `by_vpu` keys matching no requested site's VPU.
    """

    site_theta: Dict[str, float]
    source_by_site: Dict[str, str]
    unused_site_keys: FrozenSet[str]
    unused_vpu_keys: FrozenSet[str]


def resolve_theta(
    sites: Iterable[str],
    *,
    default: float = DEFAULT_THETA,
    by_vpu: Optional[Mapping[str, float]] = None,
    per_site: Optional[Mapping[str, float]] = None,
    site_vpu: Optional[Mapping[str, str]] = None,
) -> ThetaResolution:
    """Exponent per site: `per_site` > `by_vpu[site_vpu[site]]` > `default`.

    Keys are compared as strings (VPU ids like `"01"` and `"03N"` keep their
    leading zeros and suffixes; site ids keep leading zeros). Unused keys are
    reported, not raised: whether they are errors is the config layer's call.

    Raises
    ------
    ValueError
        If `default` or any supplied value is not finite and > 0.
    """
    d = check_theta(default, "theta default")
    vpu_map = {str(k): check_theta(v, f"theta.by_vpu[{k!r}]") for k, v in (by_vpu or {}).items()}
    site_map = {str(k): check_theta(v, f"theta per-site[{k!r}]") for k, v in (per_site or {}).items()}
    gage_vpu = {str(k): str(v) for k, v in (site_vpu or {}).items()}

    theta: Dict[str, float] = {}
    source: Dict[str, str] = {}
    used_vpus = set()
    for s in (str(x) for x in sites):
        vpu = gage_vpu.get(s)
        if vpu is not None:
            used_vpus.add(vpu)
        if s in site_map:
            theta[s], source[s] = site_map[s], "site"
        elif vpu is not None and vpu in vpu_map:
            theta[s], source[s] = vpu_map[vpu], "vpu"
        else:
            theta[s], source[s] = d, "default"
    return ThetaResolution(
        site_theta=theta,
        source_by_site=source,
        unused_site_keys=frozenset(site_map) - frozenset(theta),
        unused_vpu_keys=frozenset(vpu_map) - frozenset(used_vpus),
    )
