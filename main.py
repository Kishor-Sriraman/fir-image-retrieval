import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture

import src.config as config
from src.dataset import load_json_image_list, load_ground_truth_h5, TripletDataset, SpatialWordDataset
from src.extractor import SpatialVisualWordExtractor, compute_features, extract_all_from_images
from src.extractor import stack, bow_counts, tfidf_l2, vlad_encode, fisher_encode
from src.models import VisualWordCNN, train_triplet_cnn, build_spatial_pyramid
from src.evaluate import evaluate_retrieval, compute_cosine_similarity

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[DEVICE] Running on {torch.cuda.get_device_name(0) if use_cuda else 'CPU'}")

    # 1. LOAD DATA UNIFORMLY
    query_imgs = load_json_image_list(config.QUERY_JSON, config.QUERY_IMG_DIR)
    db_imgs = load_json_image_list(config.DATABASE_JSON, config.DATABASE_IMG_DIR)
    gt_matrix = load_ground_truth_h5(config.GT_FILE, key="fov")

    cache_tag = "lite" if config.USE_LITE else "full"
    
    # 2. SHARED RAW FEATURE EXTRACTION
    m_raw = extract_all_from_images(db_imgs, os.path.join(config.CACHE_DIR, f"map_raw_{cache_tag}.pkl"))
    q_raw = extract_all_from_images(query_imgs, os.path.join(config.CACHE_DIR, f"query_raw_{cache_tag}.pkl"))

    # 3. PCA DIMENSIONALITY REDUCTION 
    pca = PCA(config.PCA_DIM, random_state=config.SEED).fit(stack(m_raw, 200_000))
    proj = lambda feats: [(pca.transform(d).astype(np.float32) if len(d) else d[:, :config.PCA_DIM], xy, s) for d, xy, s in feats]
    m_f, q_f = proj(m_raw), proj(q_raw)
    sample = stack(m_f, 200_000)

    results_summary = []
    model_sims = {}
    model_orders = {}

    def register_model(sim_mat, name):
        """Uniform evaluation using your continuous evaluate_retrieval function."""
        metrics = evaluate_retrieval(sim_mat, gt_matrix, k_values=config.K_LIST, mask_self=True)
        # Store ordered indices for potential reranking
        orders = [np.argsort(sim_mat[q_idx])[::-1] for q_idx in range(sim_mat.shape[0])]
        
        row = {"method": name, **metrics}
        results_summary.append(row)
        model_sims[name] = sim_mat
        model_orders[name] = orders
        print(f"[{name}] mAP={metrics.get('mAP', 0):.4f}  nDCG@5={metrics.get('nDCG@5', 0):.4f}")

    print("\n--- Running Hybrid & Spatial Models ---")
    extractor = SpatialVisualWordExtractor(n_clusters=config.K_WORDS, grid_size=config.GRID_SIZE)
    extractor.fit_vocabulary(db_imgs, cache_path=config.KMEANS_CACHE)

    q_hists, q_grids = compute_features(extractor, query_imgs, os.path.join(config.CACHE_DIR, f"q_h_{cache_tag}.npy"), os.path.join(config.CACHE_DIR, f"q_g_{cache_tag}.npy"), "Queries")
    db_hists, db_grids = compute_features(extractor, db_imgs, os.path.join(config.CACHE_DIR, f"db_h_{cache_tag}.npy"), os.path.join(config.CACHE_DIR, f"db_g_{cache_tag}.npy"), "Database")

    extractor.compute_idf(db_grids)
    q_grids_tfidf = extractor.apply_tfidf(q_grids)
    db_grids_tfidf = extractor.apply_tfidf(db_grids)

    # NATHAN's MODELS (A - C) -----------------------------------------------------------------------------------------

    # Model A: Global BoVW
    idf_bovw = np.log((db_hists.shape[0] + 1.0) / ((db_hists > 0).sum(axis=0) + 1e-5))
    q_bovw = (q_hists / (q_hists.sum(axis=1, keepdims=True) + 1e-8)) * idf_bovw
    db_bovw = (db_hists / (db_hists.sum(axis=1, keepdims=True) + 1e-8)) * idf_bovw
    register_model(compute_cosine_similarity(q_bovw, db_bovw), "Model A (Global BoVW)")

    # Model B: Spatial Pyramid Matching (SPM)
    q_spm = build_spatial_pyramid(q_grids_tfidf)
    db_spm = build_spatial_pyramid(db_grids_tfidf)
    register_model(compute_cosine_similarity(q_spm, db_spm), "Model B (Spatial Pyramid)")

    # Model C: VisualWordCNN
    triplet_ds = TripletDataset(q_grids_tfidf, db_grids_tfidf, gt_matrix)
    if len(triplet_ds) > 0:
        triplet_loader = DataLoader(triplet_ds, batch_size=32, shuffle=True, pin_memory=use_cuda)
        cnn_model = VisualWordCNN(in_channels=config.K_WORDS, embedding_dim=256).to(device)
        cnn_model = train_triplet_cnn(cnn_model, triplet_loader, device, epochs=15, lr=1e-3)

        cnn_model.eval()
        q_loader = DataLoader(SpatialWordDataset(q_grids_tfidf), batch_size=32, shuffle=False)
        db_loader = DataLoader(SpatialWordDataset(db_grids_tfidf), batch_size=32, shuffle=False)

        def extract_embs(loader):
            embs = []
            with torch.no_grad():
                for batch_grids, _ in loader:
                    embs.append(cnn_model(batch_grids.to(device, non_blocking=True)).cpu().numpy())
            return np.concatenate(embs, axis=0)

        register_model(np.dot(extract_embs(q_loader), extract_embs(db_loader).T), "Model C (VisualWordCNN)")

    # KISHOR's MODELS (M0 - M5) ----------------------------------------------------------------------------------------------
    print("\n--- Running Partner's Encoding Models ---")
    def kmeans(k):
        return MiniBatchKMeans(k, random_state=config.SEED, n_init=3, batch_size=4096).fit(sample).cluster_centers_.astype(np.float32)

    # M0: BoW with z-score & L2 distance
    cen_bow = kmeans(config.BOW_K)
    Hm, Hq = bow_counts(m_f, cen_bow), bow_counts(q_f, cen_bow)
    mu, sd = Hm.mean(0), Hm.std(0) + 1e-9
    Zm, Zq = (Hm - mu) / sd, (Hq - mu) / sd
    S_m0 = -np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] - 2 * Zq @ Zm.T + (Zm ** 2).sum(1)[None], 0))
    register_model(S_m0, "M0 BoW (z+L2dist)")

    # M1: BoW TF-IDF
    Tm, Tq = tfidf_l2(Hm, Hq)
    register_model(Tq @ Tm.T, "M1 BoW TF-IDF")

    # M2: Soft BoW
    Sm, Sq = bow_counts(m_f, cen_bow, config.SOFT_N), bow_counts(q_f, cen_bow, config.SOFT_N)
    Tm, Tq = tfidf_l2(Sm, Sq)
    register_model(Tq @ Tm.T, f"M2 BoW Soft (n={config.SOFT_N})")

    # M3: VLAD
    cen_v = kmeans(config.VLAD_K)
    Vm, Vq = vlad_encode(m_f, cen_v), vlad_encode(q_f, cen_v)
    register_model(Vq @ Vm.T, "M3 VLAD")

    # M4: Spatial VLAD
    SVm, SVq = vlad_encode(m_f, cen_v, config.GRID_LEVELS), vlad_encode(q_f, cen_v, config.GRID_LEVELS)
    register_model(SVq @ SVm.T, "M4 Spatial VLAD")
    del SVm, SVq

    # M5: Fisher Vector
    gmm = GaussianMixture(config.FISHER_K, covariance_type="diag", max_iter=50, random_state=config.SEED).fit(sample[:100_000])
    Fm, Fq = fisher_encode(m_f, gmm), fisher_encode(q_f, gmm)
    register_model(Fq @ Fm.T, "M5 Fisher Vector")

    # FINAL RESULTS EXPORT
    df = pd.DataFrame(results_summary)
    df.to_csv(os.path.join(config.OUT_DIR, "unified_results.csv"), index=False)
    print("\n" + "=" * 65)
    print(f" UNIFIED EXPERIMENTAL RESULTS COMPARISON ({cache_tag.upper()} DATASET)")
    print("=" * 65)
    print(df.round(4).to_string(index=False))
    print("=" * 65)

if __name__ == "__main__":
    main()