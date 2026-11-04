import numpy as np
import time
import src.config as config

def nDCG(retrieved_indices, fov_gt_row, k=5):
    """Computes normalised Discounted Cumulative Gain at K for a single query using continuous FOV relevance scores."""
    top_k_indices = retrieved_indices[:k]
    gains = fov_gt_row[top_k_indices]
    discounts = np.log2(np.arange(2, k + 2))
    
    dcg = np.sum(gains / discounts)
    idcg = np.sum(np.sort(fov_gt_row)[::-1][:k] / discounts)
    
    return float(dcg / idcg) if idcg > 0 else 0.0


def average_precision(retrieved_indices, fov_gt_row, relevance_thresh=0.0):
    """
    Computes Average Precision (AP) for a single query.
    Binarizes continuous FOV scores based on `relevance_thresh`.
    """
    relevant = fov_gt_row > relevance_thresh
    
    if not np.any(relevant):
        return 0.0
        
    num_relevant = np.sum(relevant)
    precisions = []
    hits = 0
    
    for i, idx in enumerate(retrieved_indices):
        if relevant[idx]:
            hits += 1
            precisions.append(hits / (i + 1))
            
    if not precisions:
        return 0.0
        
    return float(np.sum(precisions) / num_relevant)


def evaluate_retrieval(similarity_matrix, ground_truth_fov_matrix, k_values=(1, 5, 10), mask_self=True, relevance_thresh=0.0):
    """Evaluates retrieval performance across queries using nDCG@K and mAP."""
    sim_mat = similarity_matrix.copy()

    if mask_self and sim_mat.shape[0] == sim_mat.shape[1]:
        np.fill_diagonal(sim_mat, -np.inf)

    metrics = {f"nDCG@{k}": [] for k in k_values}
    metrics["mAP"] = []
    
    for q_idx in range(sim_mat.shape[0]):
        retrieved = np.argsort(sim_mat[q_idx])[::-1]
        fov_gt_row = ground_truth_fov_matrix[q_idx]
        
        for k in k_values:
            metrics[f"nDCG@{k}"].append(nDCG(retrieved, fov_gt_row, k=k))
            
        metrics["mAP"].append(average_precision(retrieved, fov_gt_row, relevance_thresh=relevance_thresh))

    return {m: float(np.mean(scores)) for m, scores in metrics.items()}


def compute_cosine_similarity(q_feats, db_feats):
    """L2-normalizes features along axis 1 and computes dot product cosine similarity."""
    q_norm = q_feats / (np.linalg.norm(q_feats, axis=1, keepdims=True) + 1e-8)
    db_norm = db_feats / (np.linalg.norm(db_feats, axis=1, keepdims=True) + 1e-8)
    return np.dot(q_norm, db_norm.T)

def evaluate(S, gt, name, enc_time, dim):

    t0 = time.time()
    order = np.argsort(-S, axis=1)
    search_t = time.time() - t0
    ap, res = [], {f"{m}@{k}": [] for k in config.K_LIST for m in ("P", "R", "F1")}
    valid = np.where(gt.sum(1) > 0)[0]
    for q in valid:
        rel = gt[q][order[q]]
        n_rel = rel.sum()
        cum = np.cumsum(rel)
        ap.append((cum[rel] / (np.where(rel)[0] + 1)).sum() / n_rel)
        for k in config.K_LIST:
            p, r = cum[k - 1] / k, cum[k - 1] / n_rel
            res[f"P@{k}"].append(p); res[f"R@{k}"].append(r)
            res[f"F1@{k}"].append(0 if p + r == 0 else 2 * p * r / (p + r))
    row = dict(method=name, dim=dim, MAP=np.mean(ap), **{k: np.mean(v) for k, v in res.items()},
               ms_per_query=1000 * (enc_time + search_t) / len(S))
    return row, np.array(ap), order

def eval_order(order, gt, name, dim, t):
    S_fake = np.zeros(order.shape)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(order.shape[1])[None, :].repeat(len(order), 0), 1)
    S_fake = -rank.astype(float)
    return evaluate(S_fake, gt, name, t, dim)
