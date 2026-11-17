"""Helpers for the scaling-DA unit tests (hand-built graphs, no I/O)."""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

import numpy as np
import pytest


def rconn_from_edges(edges: Mapping[int, int]) -> Dict[int, List[int]]:
    """Reverse connectivity from `{upstream: downstream}`; ids sorted for determinism."""
    r: Dict[int, List[int]] = {}
    for u, d in sorted(edges.items()):
        r.setdefault(d, []).append(u)
        r.setdefault(u, [])
    return r


def total_area(edges: Mapping[int, int], incr: Mapping[int, float]) -> Dict[int, float]:
    """Accumulate incremental areas downstream into total drainage areas."""
    r = rconn_from_edges(edges)
    tot: Dict[int, float] = {}

    def acc(s: int) -> float:
        if s not in tot:
            tot[s] = incr[s] + sum(acc(u) for u in r.get(s, ()))
        return tot[s]

    for s in r:
        acc(s)
    return tot


def random_network(rng: np.random.Generator, n: int) -> Tuple[Dict[int, int], Dict[int, float]]:
    """Random tree network: node 0 is the outlet, node i>0 drains to a random j<i.

    Returns `(edges upstream->downstream, total area)`. Ids are offset by 1000
    so they never coincide with row indices.
    """
    edges = {1000 + i: 1000 + int(rng.integers(0, i)) for i in range(1, n)}
    incr = {1000 + i: float(rng.uniform(0.5, 20.0)) for i in range(n)}
    return edges, total_area(edges, incr)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "hfv4_fixture: needs an HFv4.0 subset gpkg at $TROUTE_HFV4_FIXTURE (generated, not committed)",
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261102)


@pytest.fixture
def g():
    """Graph helpers as a fixture (avoids importing conftest under any import mode)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        rconn_from_edges=rconn_from_edges, total_area=total_area, random_network=random_network
    )
