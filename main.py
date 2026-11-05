import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul")
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

import src.config as config
from src.dataset import load_json_image_list, load_ground_truth_h5, TripletDataset
from src.evaluate import evaluate_all, order_to_scores, compute_cosine_similarity
from src.extractor import (extract_all_from_images, stack, bow_counts, tfidf_l2, vlad_encode, fisher_encode,
                           spatial_word_grids, grid_idf, grid_tfidf)
from src.models import VisualWordCNN, train_triplet_cnn, build_spatial_pyramid
from src.rerank import geometric_rerank

RANDOM = "Random ranking"
M0 = "M0 BoW (z+L2dist)"
GV = f"M6 Best+GV(top{config.GV_SHORTLIST})"
MODEL_C = "Model C (VisualWordCNN, CV)"

METRIC_COLS = (["MAP"] + [f"{m}@{k}" for k in config.K_LIST for m in ("P", "R", "F1", "nDCG")] + ["ms_per_query"])


# ----------------------------------------------------------------------------------------------
# Model C: learned embedding on the spatial visual-word grids, scored with cross-validation
# ----------------------------------------------------------------------------------------------
def embed(model, grids, device, batch_size=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(grids), batch_size):
            x = torch.from_numpy(grids[i:i + batch_size]).float().to(device)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out, axis=0)


def run_model_c(q_grids, db_grids, gt_fov, device, seed):
    """Returns (S, embed_seconds). Row q of S comes from a model trained on the OTHER folds only."""
    use_cuda = device.type == "cuda"
    n_q = len(q_grids)
    S = np.zeros((n_q, len(db_grids)), dtype=np.float32)
    embed_time = 0.0
    kf = KFold(config.N_FOLDS, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_q))):
        fold_seed = seed * 100 + fold
        torch.manual_seed(fold_seed)
        if use_cuda:
            torch.cuda.manual_seed_all(fold_seed)

        ds = TripletDataset(q_grids[train_idx], db_grids, gt_fov[train_idx], seed=fold_seed)
        if len(ds) == 0:
            raise RuntimeError(f"No training triplets in fold {fold}; check the fov thresholds in TripletDataset.")
        loader = DataLoader(ds, batch_size=config.CNN_BATCH, shuffle=True, drop_last=True, pin_memory=use_cuda)

        print(f"[Model C] fold {fold + 1}/{config.N_FOLDS}: train queries={len(train_idx)}, test queries={len(test_idx)}")
        model = VisualWordCNN(in_channels=config.K_WORDS, embedding_dim=config.CNN_EMB_DIM).to(device)
        model = train_triplet_cnn(model, loader, device, epochs=config.CNN_EPOCHS, lr=config.CNN_LR)
        model.eval()

        t0 = time.time()
        db_emb = embed(model, db_grids, device)
        q_emb = embed(model, q_grids[test_idx], device)
        S[test_idx] = q_emb @ db_emb.T
        embed_time += (time.time() - t0)
    # every fold embeds the whole database once, so report the average time of one full pass
    return S, embed_time / config.N_FOLDS


# ----------------------------------------------------------------------------------------------
# One full pass over all models for one seed
# ----------------------------------------------------------------------------------------------
def run_seed(seed, m_raw, q_raw, gt_sim, gt_fov, device):
    rows, aps, orders = [], {}, {}
    n_q = gt_sim.shape[0]

    def register(S, name, enc_time, dim, K=None, **extra):
        row, ap, order = evaluate_all(S, gt_sim, gt_fov, name, enc_time, dim, K=K, seed=seed, **extra)
        rows.append(row)
        aps[name] = ap
        orders[name] = order
        print(f"[seed {seed}] {name:<30} MAP={row['MAP']:.4f}  P@10={row['P@10']:.4f}  "
              f"nDCG@10={row['nDCG@10']:.4f}  dim={dim}")
        return row

    # --- shared front end: PCA on local descriptors, shared vocabulary ----------------------------
    pca = PCA(config.PCA_DIM, random_state=seed).fit(stack(m_raw, 200_000, seed))
    proj = lambda feats: [(pca.transform(d).astype(np.float32) if len(d) else d[:, :config.PCA_DIM], xy, s)
                          for d, xy, s in feats]
    m_f, q_f = proj(m_raw), proj(q_raw)
    sample = stack(m_f, 200_000, seed)

    def kmeans(k):
        return MiniBatchKMeans(k, random_state=seed, n_init=3, batch_size=4096).fit(sample).cluster_centers_.astype(np.float32)

    # --- chance level ------------------------------------------------------------------------------
    register(np.random.default_rng(seed).random(gt_sim.shape), RANDOM, 0.0, 0)

    # --- vocabulary shared by BoW (M0-M2) and the spatial models (A-C) -----------------------------
    cen_bow = kmeans(config.BOW_K)
    cen_words = cen_bow if config.K_WORDS == config.BOW_K else kmeans(config.K_WORDS)

    # --- Nathan's models (A - C) on the shared front end ---------------------------------------------
    t0 = time.time()
    q_grids = spatial_word_grids(q_f, cen_words, config.GRID_SIZE)
    db_grids = spatial_word_grids(m_f, cen_words, config.GRID_SIZE)
    t_grid = time.time() - t0
    idf = grid_idf(db_grids)
    q_tf, db_tf = grid_tfidf(q_grids, idf), grid_tfidf(db_grids, idf)

    # Model A: global BoVW (L1 + IDF, cosine)
    t0 = time.time()
    q_h, db_h = q_grids.sum(axis=(2, 3)), db_grids.sum(axis=(2, 3))
    idf_b = np.log((db_h.shape[0] + 1.0) / ((db_h > 0).sum(axis=0) + 1e-5))
    q_b = (q_h / (q_h.sum(axis=1, keepdims=True) + 1e-8)) * idf_b
    db_b = (db_h / (db_h.sum(axis=1, keepdims=True) + 1e-8)) * idf_b
    S = compute_cosine_similarity(q_b, db_b)
    register(S, "Model A (Global BoVW)", t_grid + time.time() - t0, config.K_WORDS, config.K_WORDS)

    # Model B: spatial pyramid matching
    t0 = time.time()
    q_spm, db_spm = build_spatial_pyramid(q_tf), build_spatial_pyramid(db_tf)
    S = compute_cosine_similarity(q_spm, db_spm)
    register(S, "Model B (Spatial Pyramid)", t_grid + time.time() - t0, q_spm.shape[1], config.K_WORDS)

    # Model C: learned embedding, cross-validated over queries
    S, t_embed = run_model_c(q_tf, db_tf, gt_fov, device, seed)
    register(S, MODEL_C, t_embed, config.CNN_EMB_DIM, config.K_WORDS)

    # --- Kishor's models (M0 - M5) -------------------------------------------------------------------
    t0 = time.time()
    Hm, Hq = bow_counts(m_f, cen_bow), bow_counts(q_f, cen_bow)
    mu, sd = Hm.mean(0), Hm.std(0) + 1e-9
    Zm, Zq = (Hm - mu) / sd, (Hq - mu) / sd
    S = -np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] - 2 * Zq @ Zm.T + (Zm ** 2).sum(1)[None], 0))
    register(S, M0, time.time() - t0, config.BOW_K, config.BOW_K)

    t0 = time.time()
    Tm, Tq = tfidf_l2(Hm, Hq)
    S = Tq @ Tm.T
    register(S, "M1 BoW TF-IDF", time.time() - t0, config.BOW_K, config.BOW_K)

    t0 = time.time()
    Sm, Sq = bow_counts(m_f, cen_bow, config.SOFT_N), bow_counts(q_f, cen_bow, config.SOFT_N)
    Tm, Tq = tfidf_l2(Sm, Sq)
    S = Tq @ Tm.T
    register(S, f"M2 BoW Soft (n={config.SOFT_N})", time.time() - t0, config.BOW_K, config.BOW_K)

    cen_v = kmeans(config.VLAD_K)
    t0 = time.time()
    Vm, Vq = vlad_encode(m_f, cen_v), vlad_encode(q_f, cen_v)
    S = Vq @ Vm.T
    register(S, "M3 VLAD", time.time() - t0, Vm.shape[1], config.VLAD_K)

    t0 = time.time()
    SVm, SVq = vlad_encode(m_f, cen_v, config.GRID_LEVELS), vlad_encode(q_f, cen_v, config.GRID_LEVELS)
    S = SVq @ SVm.T
    register(S, "M4 Spatial VLAD", time.time() - t0, SVm.shape[1], config.VLAD_K)
    del SVm, SVq

    gmm = GaussianMixture(config.FISHER_K, covariance_type="diag", max_iter=50, random_state=seed).fit(sample[:100_000])
    t0 = time.time()
    Fm, Fq = fisher_encode(m_f, gmm), fisher_encode(q_f, gmm)
    S = Fq @ Fm.T
    register(S, "M5 Fisher Vector", time.time() - t0, Fm.shape[1], config.FISHER_K)

    # --- M6: geometric verification on top of the best method ------------------------------------------
    best = max((r for r in rows if r["method"] != RANDOM), key=lambda r: r["MAP"])
    t0 = time.time()
    new_order = geometric_rerank(orders[best["method"]], q_raw, m_raw, config.GV_SHORTLIST)
    gv_time = time.time() - t0 + best["ms_per_query"] * n_q / 1000.0
    register(order_to_scores(new_order), GV, gv_time, best["dim"], best["K"], note=f"GV on {best['method']}")

    return rows, aps


# ----------------------------------------------------------------------------------------------
# Aggregation, significance tests, plot
# ----------------------------------------------------------------------------------------------
def aggregate_and_export(all_rows, all_aps, n_valid, tag):
    df_seed = pd.DataFrame(all_rows)
    df_seed.to_csv(os.path.join(config.OUT_DIR, "results_per_seed.csv"), index=False)

    g = df_seed.groupby("method", sort=False)
    meta = g[["dim", "K"] + (["note"] if "note" in df_seed else [])].first()
    mean, std = g[METRIC_COLS].mean(), g[METRIC_COLS].std().fillna(0.0)
    out = meta.join(mean)
    out["MAP_std"] = std["MAP"]
    out.insert(0, "descriptor", config.DESCRIPTOR)
    out.insert(1, "keypoints", config.N_KEYPOINTS)
    out.insert(2, "pca", config.PCA_DIM)
    out = out.reset_index()
    out.to_csv(os.path.join(config.OUT_DIR, "unified_results.csv"), index=False)

    # per-query AP averaged over seeds -> paired Wilcoxon tests (all methods share the same valid queries)
    names = list(out["method"])
    mean_ap = {n: np.mean([aps[n] for aps in all_aps], axis=0) for n in names}
    ranked = out[out["method"] != RANDOM].sort_values("MAP", ascending=False)["method"].tolist()
    pairs = [(n, ref) for ref in (M0, RANDOM) for n in names if n != ref]
    if len(ranked) >= 2:
        pairs.append((ranked[0], ranked[1]))          # best two methods against each other
        extra_pair = (MODEL_C, "M5 Fisher Vector")        # learned model vs best encoding
        if extra_pair[0] in names and extra_pair[1] in names:
            pairs.append(extra_pair)
    wrows, seen = [], set()
    for n, ref in pairs:
        if (n, ref) in seen:
            continue
        seen.add((n, ref))
        diff = mean_ap[n] - mean_ap[ref]
        p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        wrows.append(dict(method=n, vs=ref, mean_dAP=diff.mean(), p_value=p, significant=bool(p < 0.05)))
    pd.DataFrame(wrows).to_csv(os.path.join(config.OUT_DIR, "wilcoxon.csv"), index=False)
    np.savez(os.path.join(config.OUT_DIR, "per_query_ap.npz"),
             **{re.sub(r"\W+", "_", n): v for n, v in mean_ap.items()})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(out["method"][::-1], out["MAP"][::-1], color="#3b6fb6")
        for i, v in enumerate(out["MAP"][::-1]):
            ax.text(v, i, f" {v:.3f}", va="center")
        ax.set_xlabel("MAP (binary sim)")
        ax.set_title(f"{config.DESCRIPTOR.upper()}, {config.N_KEYPOINTS} keypoints, PCA {config.PCA_DIM}, {tag} dataset")
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUT_DIR, "results.png"), dpi=150)
    except Exception as e:
        print("plot skipped:", e)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 100)
    print(f" UNIFIED RESULTS ({tag.upper()}), {len(config.SEEDS)} seed(s), {n_valid} queries with >= 1 relevant image")
    print("=" * 100)
    print(out.round(4).to_string(index=False))
    print("\nWilcoxon (paired, per-query AP):")
    print(pd.DataFrame(wrows).round(4).to_string(index=False))


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[DEVICE] Running on {torch.cuda.get_device_name(0) if use_cuda else 'CPU'}")
    assert config.GRID_SIZE == (12, 12), "build_spatial_pyramid is written for a 12x12 grid"

    # 1. LOAD DATA (fails loudly on unreadable images or mismatching ground truth)
    query_imgs = load_json_image_list(config.QUERY_JSON, config.QUERY_IMG_DIR)
    db_imgs = load_json_image_list(config.DATABASE_JSON, config.DATABASE_IMG_DIR)
    gt_sim = load_ground_truth_h5(config.GT_FILE, key="sim", as_bool=True)   # binary: MAP, P/R/F1, Wilcoxon
    gt_fov = load_ground_truth_h5(config.GT_FILE, key="fov")                 # graded: nDCG only
    assert gt_sim.shape == gt_fov.shape == (len(query_imgs), len(db_imgs)), \
        (gt_sim.shape, gt_fov.shape, len(query_imgs), len(db_imgs))

    # The cache name contains every setting that changes the features, so old caches are never reused by mistake.
    mode = "lite" if config.USE_LITE else "full"
    tag = f"{mode}_{config.DESCRIPTOR}_{config.N_KEYPOINTS}_{config.MAX_SIDE}"
    m_raw = extract_all_from_images(db_imgs, os.path.join(config.CACHE_DIR, f"map_raw_{tag}.pkl"))
    q_raw = extract_all_from_images(query_imgs, os.path.join(config.CACHE_DIR, f"query_raw_{tag}.pkl"))
    del query_imgs, db_imgs
    print("avg keypoints/image (database):", np.mean([len(f[0]) for f in m_raw]).round(1))

    # 2. RUN ALL MODELS FOR EVERY SEED
    all_rows, all_aps = [], []
    for seed in config.SEEDS:
        print(f"\n===== SEED {seed} =====")
        rows, aps = run_seed(seed, m_raw, q_raw, gt_sim, gt_fov, device)
        all_rows += rows
        all_aps.append(aps)

    # 3. EXPORT
    aggregate_and_export(all_rows, all_aps, int((gt_sim.sum(1) > 0).sum()), mode)


if __name__ == "__main__":
    main()