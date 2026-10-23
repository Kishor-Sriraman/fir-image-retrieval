import numpy as np

def nDCG(retrieved_indices, fov_gt_row, k=5):
    """Computes normalised Discounted Cumulative Gain at K for a single query using continuous FOV relevance scores instead of the binary scores."""
    top_k_indices = retrieved_indices[:k]
    gains = fov_gt_row[top_k_indices]
    discounts = np.log2(np.arange(2, k + 2))
    
    dcg = np.sum(gains / discounts)
    idcg = np.sum(np.sort(fov_gt_row)[::-1][:k] / discounts)
    
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_retrieval(similarity_matrix, ground_truth_fov_matrix, k_values=(1, 5, 10), mask_self=True):
    """Evaluates retrieval performance across queries using nDCG@K."""
    sim_mat = similarity_matrix.copy()

    if mask_self and sim_mat.shape[0] == sim_mat.shape[1]:
        np.fill_diagonal(sim_mat, -np.inf)

    metrics = {f"nDCG@{k}": [] for k in k_values}
    for q_idx in range(sim_mat.shape[0]):
        retrieved = np.argsort(sim_mat[q_idx])[::-1]
        fov_gt_row = ground_truth_fov_matrix[q_idx]
        for k in k_values:
            metrics[f"nDCG@{k}"].append(nDCG(retrieved, fov_gt_row, k=k))

    return {m: float(np.mean(scores)) for m, scores in metrics.items()}


def compute_cosine_similarity(q_feats, db_feats):
    """L2-normalizes features along axis 1 and computes dot product cosine similarity."""
    q_norm = q_feats / (np.linalg.norm(q_feats, axis=1, keepdims=True) + 1e-8)
    db_norm = db_feats / (np.linalg.norm(db_feats, axis=1, keepdims=True) + 1e-8)
    return np.dot(q_norm, db_norm.T)