"""Graph utilities for spatial modelling."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def build_adjacency_matrix(
    nodes: Sequence[str],
    edges: Optional[Iterable[Sequence[str]]] = None,
    fully_connected: bool = True,
    include_self_loops: bool = True,
) -> np.ndarray:
    """Create an adjacency matrix for the provided nodes."""

    node_list: List[str] = list(nodes)
    index: Dict[str, int] = {name: idx for idx, name in enumerate(node_list)}
    size = len(node_list)
    adjacency = np.zeros((size, size), dtype=np.float32)

    if fully_connected:
        adjacency[:] = 1.0
    if include_self_loops:
        np.fill_diagonal(adjacency, 1.0)

    if edges is not None:
        adjacency[:] = 0.0
        if include_self_loops:
            np.fill_diagonal(adjacency, 1.0)
        for edge in edges:
            if len(edge) < 2:
                continue
            src, dst = edge[0], edge[1]
            if src not in index or dst not in index:
                continue
            adjacency[index[src], index[dst]] = 1.0
            adjacency[index[dst], index[src]] = 1.0

    return adjacency


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Symmetrically normalise an adjacency matrix."""

    adjacency = np.asarray(adjacency, dtype=np.float32)
    degree = adjacency.sum(axis=1)
    degree[degree == 0.0] = 1.0
    inv_sqrt_degree = 1.0 / np.sqrt(degree)
    inv_sqrt = np.diag(inv_sqrt_degree)
    return inv_sqrt @ adjacency @ inv_sqrt

