from __future__ import annotations

from typing import Any

import numpy as np
import networkx as nx

from .graph import symmetrized_upper_edges


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    unique, inverse = np.unique(labels, return_inverse=True)
    # Sort clusters by decreasing size for stable labels.
    counts = np.bincount(inverse)
    order = np.argsort(-counts)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[inverse].astype(np.int64)


def cluster_item_graph_leiden(
    A,
    resolution: float = 1.0,
    seed: int = 123,
    fallback: str = "louvain",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster the item competition graph using Leiden when available.

    The implementation tries ``python-igraph`` + ``leidenalg`` first. If they are
    not installed, the default fallback is NetworkX Louvain so that smoke tests
    remain runnable. For final experiments, install the optional Leiden
    dependencies and check ``cluster_method`` in the saved metadata.
    """
    n = A.shape[0]
    src, dst, w = symmetrized_upper_edges(A)
    if len(src) == 0:
        labels = np.arange(n, dtype=np.int64)
        info = {
            "cluster_method": "singleton_no_edges",
            "resolution": float(resolution),
            "n_clusters": int(n),
            "cluster_size_min": 1,
            "cluster_size_median": 1.0,
            "cluster_size_max": 1,
            "fallback": fallback,
        }
        return labels, info

    method = "leiden"
    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore

        g = ig.Graph(n=int(n), edges=list(zip(src.tolist(), dst.tolist())), directed=False)
        g.es["weight"] = w.tolist()
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(resolution),
            seed=int(seed),
        )
        labels = np.asarray(part.membership, dtype=np.int64)
    except Exception as exc:
        if fallback.lower() not in {"louvain", "greedy", "none"}:
            raise
        if fallback.lower() == "none":
            raise RuntimeError(
                "Leiden clustering requires python-igraph and leidenalg. Install optional dependencies or use --cluster-fallback louvain."
            ) from exc
        G = nx.Graph()
        G.add_nodes_from(range(int(n)))
        G.add_weighted_edges_from((int(i), int(j), float(x)) for i, j, x in zip(src, dst, w))
        if fallback.lower() == "louvain" and hasattr(nx.community, "louvain_communities"):
            communities = nx.community.louvain_communities(
                G,
                weight="weight",
                resolution=float(resolution),
                seed=int(seed),
            )
            method = "networkx_louvain_fallback"
        else:
            communities = nx.community.greedy_modularity_communities(G, weight="weight")
            method = "networkx_greedy_fallback"
        labels = np.empty(int(n), dtype=np.int64)
        for k, nodes in enumerate(communities):
            labels[list(nodes)] = int(k)

    labels = _relabel_consecutive(labels)
    counts = np.bincount(labels)
    info = {
        "cluster_method": method,
        "resolution": float(resolution),
        "n_clusters": int(len(counts)),
        "cluster_size_min": int(counts.min()) if len(counts) else 0,
        "cluster_size_median": float(np.median(counts)) if len(counts) else 0.0,
        "cluster_size_max": int(counts.max()) if len(counts) else 0,
        "fallback": fallback,
        "n_edges_undirected": int(len(src)),
    }
    return labels.astype(np.int64), info
