from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def ensure_csr(A) -> sparse.csr_matrix:
    """Return a float32 CSR matrix for graph computations."""
    if sparse.issparse(A):
        return A.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(A, dtype=np.float32))


def topk_competition_graph(
    emb: np.ndarray,
    k: int = 10,
    symmetrize: bool = True,
    row_normalize: bool = True,
    return_sparse: bool = False,
):
    """Build a weighted top-k item competition graph from cosine similarities.

    The graph keeps the top-k cosine neighbors for each item. By default the
    returned graph is dense for backward compatibility; pass ``return_sparse=True``
    to return a CSR matrix, which is preferred for 10k+ item simulations.
    """
    n = emb.shape[0]
    emb = normalize(emb, norm="l2", axis=1)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine", algorithm="auto")
    nn.fit(emb)
    dist, idx = nn.kneighbors(emb)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n):
        for d, j in zip(dist[i], idx[i]):
            if i == j:
                continue
            sim = max(0.0, 1.0 - float(d))
            if sim > 0:
                rows.append(i)
                cols.append(int(j))
                vals.append(sim)
    A = sparse.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(n, n), dtype=np.float32)
    if symmetrize:
        A = A.maximum(A.T)
    if row_normalize:
        row_sum = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
        row_sum[row_sum == 0] = 1.0
        A = sparse.diags(1.0 / row_sum, format="csr") @ A
        A = A.astype(np.float32)
    return A if return_sparse else A.toarray().astype(np.float32)


def graph_edges(A):
    """Return edge index and weights for nonzero adjacency entries.

    Supports both dense NumPy arrays and SciPy sparse matrices.
    """
    if sparse.issparse(A):
        coo = A.tocoo()
        src = coo.row
        dst = coo.col
        w = coo.data
    else:
        src, dst = np.nonzero(A)
        w = np.asarray(A)[src, dst]
    mask = src != dst
    return src[mask].astype(np.int64), dst[mask].astype(np.int64), np.asarray(w[mask], dtype=np.float32)


def symmetrized_upper_edges(A):
    """Return undirected upper-triangular graph edges and weights for clustering."""
    A_csr = ensure_csr(A)
    # Use maximum instead of average because row normalization makes A asymmetric.
    B = A_csr.maximum(A_csr.T)
    B.setdiag(0.0)
    B.eliminate_zeros()
    U = sparse.triu(B, k=1).tocoo()
    return U.row.astype(np.int64), U.col.astype(np.int64), U.data.astype(np.float32)
