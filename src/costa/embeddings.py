from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def cf_item_embedding(user_item: sparse.csr_matrix, dim: int = 64, random_state: int = 123) -> np.ndarray:
    """Compute item embeddings from a sparse user-item implicit matrix.

    We apply log1p weighting and truncated SVD to the item-by-user matrix.
    The returned matrix has shape (n_items, dim) and unit-normalized rows.
    """
    if not sparse.isspmatrix_csr(user_item):
        user_item = user_item.tocsr()
    x = user_item.copy().astype(float)
    x.data = np.log1p(x.data)
    item_user = x.T.tocsr()
    dim_eff = min(dim, max(2, min(item_user.shape) - 1))
    svd = TruncatedSVD(n_components=dim_eff, random_state=random_state)
    emb = svd.fit_transform(item_user)
    emb = normalize(emb, norm="l2", axis=1)
    return emb.astype(np.float32)


def hybrid_embedding(cf_emb: np.ndarray, feature_emb: np.ndarray | None = None, cf_weight: float = 0.8) -> np.ndarray:
    """Optionally combine CF embeddings with item content/property embeddings."""
    if feature_emb is None:
        return cf_emb
    cf_weight = float(cf_weight)
    x = np.concatenate([cf_weight * cf_emb, (1.0 - cf_weight) * feature_emb], axis=1)
    return normalize(x, norm="l2", axis=1).astype(np.float32)
