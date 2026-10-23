import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import src.config as config
from src.dataset import load_json_image_list, load_ground_truth_h5, TripletDataset, SpatialWordDataset
from src.extractor import SpatialVisualWordExtractor, compute_features
from src.models import VisualWordCNN, train_triplet_cnn, build_spatial_pyramid
from src.evaluate import nDCG, evaluate_retrieval, compute_cosine_similarity

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[DEVICE] Running on {torch.cuda.get_device_name(0) if use_cuda else 'CPU'}")

    # Load Data and ground truth
    query_imgs = load_json_image_list(config.QUERY_JSON, config.QUERY_IMG_DIR)
    db_imgs = load_json_image_list(config.DATABASE_JSON, config.DATABASE_IMG_DIR)
    gt_matrix = load_ground_truth_h5(config.GT_FILE, key="fov")

    # Cache Setup
    mode = "lite" if config.USE_LITE else "full" # Use small dataset or the entire one
    cache = lambda name: os.path.join(config.CACHE_DIR, f"{name}_{mode}.npy")

    # Extract ORB keypoint features  --> cluster with k-means --> turn them  into visual words
    extractor = SpatialVisualWordExtractor(n_clusters=config.K_WORDS, grid_size=config.GRID_SIZE)
    # Fit on database for shared visual vocabular
    extractor.fit_vocabulary(db_imgs, cache_path=config.KMEANS_CACHE) 

    # Compute 1D histogram for model A, and 3D spatial grids for model B & C.
    q_hists, q_grids = compute_features(extractor, query_imgs, cache("query_hists"), cache("query_grids"), "Queries") 
    db_hists, db_grids = compute_features(extractor, db_imgs, cache("db_hists"), cache("db_grids"), "Database")

    # Compute IDF weights on 3D database grids, then apply term frequency to both query and database grids to match vector scaling (also avoid data leakage)
    extractor.compute_idf(db_grids) 
    q_grids_tfidf = extractor.apply_tfidf(q_grids)
    db_grids_tfidf = extractor.apply_tfidf(db_grids)

    # MODEL A: Global BoVW Baseline (Ignores spatial coordinates)
    # Calculate 'normal' TF and IDF at whole-image lavel, based on histogram, and computes result using cosine similarity.
    print("\n Evaluating Model A: Classical Global BoVW")
    idf_bovw = np.log((db_hists.shape[0] + 1.0) / ((db_hists > 0).sum(axis=0) + 1e-5))
    q_bovw_tfidf = (q_hists / (q_hists.sum(axis=1, keepdims=True) + 1e-8)) * idf_bovw
    db_bovw_tfidf = (db_hists / (db_hists.sum(axis=1, keepdims=True) + 1e-8)) * idf_bovw
    res_a = evaluate_retrieval(compute_cosine_similarity(q_bovw_tfidf, db_bovw_tfidf), gt_matrix)

    # MODEL B: Direct Spatial TF-IDF Grid
    # Flatten 3D spatial TF-IDF grids, into 1D vectors to retain spatial structure without machine learning.
    print("[2/3] Evaluating Model B: Spatial Pyramid Matching (SPM)...")
    # Build hyrarchical spatial pyramid features
    q_spm = build_spatial_pyramid(q_grids_tfidf)
    db_spm = build_spatial_pyramid(db_grids_tfidf)
    res_b = evaluate_retrieval(compute_cosine_similarity(q_spm, db_spm), gt_matrix)
    
    # MODEL C: VisualWordCNN
    # CNN model on the 3D TF-IDF grid tensors.
    print("[3/3] Evaluating Model C: Fine-Tuned VisualWordCNN...")

    # Find anchor, positive, and negative points to train triplets on 
    triplet_ds = TripletDataset(q_grids_tfidf, db_grids_tfidf, gt_matrix)

    if len(triplet_ds) > 0:
        # Load training triplets and CNN architecture
        triplet_loader = DataLoader(triplet_ds, batch_size=32, shuffle=True, pin_memory=use_cuda)
        model = VisualWordCNN(in_channels=config.K_WORDS, embedding_dim=256).to(device)
        model = train_triplet_cnn(model, triplet_loader, device, epochs=15, lr=1e-3)

        model.eval()
        q_loader = DataLoader(SpatialWordDataset(q_grids_tfidf), batch_size=32, shuffle=False)
        db_loader = DataLoader(SpatialWordDataset(db_grids_tfidf), batch_size=32, shuffle=False)

        # Batch feature extraction to collect embeddings
        def extract_embs(loader):
            embs = []
            with torch.no_grad():
                for batch_grids, _ in loader:
                    embs.append(model(batch_grids.to(device, non_blocking=True)).cpu().numpy())
            return np.concatenate(embs, axis=0)

        # Normalized embeddings are extracted, so np.dot directly equals Cosine similarity
        res_c = evaluate_retrieval(np.dot(extract_embs(q_loader), extract_embs(db_loader).T), gt_matrix) 
    else:
        print("[WARNING] No valid triplets found. Skipping Model C.")
        res_c = None


    # Benchmark Results Summary
    print("\n" + "=" * 65)
    print(f"   EXPERIMENTAL RESULTS COMPARISON ({'LITE' if config.USE_LITE else 'FULL'} DATASET)")
    print("=" * 65)
    print(f"{'Metric':<15} | {'Model A (BoVW)':<15} | {'Model B (Spatial)':<18} | {'Model C (CNN)':<15}")
    print("-" * 65)

    for m in ["nDCG@1", "nDCG@5", "nDCG@10"]:
        val_a, val_b = res_a.get(m, 0.0), res_b.get(m, 0.0)
        val_c = res_c.get(m, 0.0) if res_c else 0.0
        print(f"{m:<15} | {val_a:<15.4f} | {val_b:<18.4f} | {val_c:<15.4f}")

    print("=" * 65)


if __name__ == "__main__":
    main()