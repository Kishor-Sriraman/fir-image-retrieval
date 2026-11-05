# Image retrieval on London street-level images (FIR)

Retrieval of relevant street-level images (query -> database) with Bag-of-Visual-Words (BoVW) and several improvements. All methods share **one front end** and **one evaluation**, and every number in the final table comes from a single run of `main.py`.


## Pipeline

1. **Local features (shared by every model):** RootSIFT, up to 2000 keypoints per image (about 660 found on average), images resized to max side 640.
2. **PCA** of the local descriptors to 64 dimensions.
3. **Visual vocabulary** with k-means (K = 128), shared by all BoW-based models.
4. **Encoding / matching step** (this is what differs between models, see below).
5. **Evaluation** of the ranking of the database for every query.

## Models

| ID | Method | Idea |
|---|---|---|
| Random | Random ranking | Chance level |
| M0 | BoW baseline | Hard-assignment counts, z-score, Euclidean distance |
| M1 | BoW TF-IDF | Log counts, IDF, L2 normalisation, cosine |
| M2 | Soft-assignment BoW | Each descriptor votes for its 3 nearest words |
| A | Global BoVW | L1 + IDF histogram from the spatial grids, cosine |
| B | Spatial pyramid | 1x1, 2x2 and 4x4 pyramid over the visual-word grids, cosine |
| C | Learned embedding (CNN) | Small CNN on the 12x12 visual-word grids, triplet loss, **5-fold cross-validated over the queries** |
| M3 | VLAD | Residuals per visual word, intra-, power- and L2-normalisation |
| M4 | Spatial VLAD | VLAD per cell of a 1x1 and 2x2 grid |
| M5 | Fisher vector | Diagonal GMM, gradients w.r.t. mean and variance |
| M6 | Best method + geometric verification | Re-ranks the top 30 results by RANSAC inliers between matched keypoints |

Model C is the only method trained on labels. Every query is scored by a model that never saw it during training.

## Evaluation

- **MAP, P@k, R@k, F1@k (k = 5, 10):** binary relevance `sim` (about 7 relevant images per query). Queries without a relevant image are skipped, so all methods use the same queries.
- **nDCG@k (k = 5, 10):** graded relevance `fov`, on the same queries.
- **Per-query AP** for paired Wilcoxon signed-rank tests (significance level 0.05).
- **Seeds:** results are the mean over 3 seeds (0, 1, 2); `MAP_std` is the standard deviation over seeds. The Wilcoxon tests use the per-query AP averaged over the seeds.
- `ms_per_query` = (time to encode all images + ranking) / number of queries. For Model C only embedding is counted, not training.

## Results (lite set: 1000 database images, 500 queries, mean over 3 seeds)

| Method | Dim | MAP | P@10 | R@10 | F1@10 | nDCG@10 | ms/query |
|---|---|---|---|---|---|---|---|
| Random | - | 0.0133 | 0.0069 | 0.0096 | 0.0076 | 0.0086 | - |
| M0 BoW (z-score + Euclidean) | 128 | 0.0514 | 0.0337 | 0.0599 | 0.0409 | 0.0572 | 0.82 |
| M1 BoW TF-IDF | 128 | 0.0699 | 0.0520 | 0.0690 | 0.0551 | 0.0684 | 0.07 |
| M2 BoW soft (n=3) | 128 | 0.0713 | 0.0512 | 0.0750 | 0.0572 | 0.0731 | 5.22 |
| A Global BoVW | 128 | 0.0586 | 0.0415 | 0.0657 | 0.0481 | 0.0596 | 1.23 |
| B Spatial pyramid | 2688 | 0.0458 | 0.0339 | 0.0481 | 0.0370 | 0.0452 | 2.02 |
| C Learned embedding (CV) | 256 | 0.1582 | 0.1250 | 0.1708 | 0.1316 | 0.1650 | 0.85 |
| M3 VLAD | 8192 | 0.1139 | 0.0767 | 0.1412 | 0.0937 | 0.1361 | 2.26 |
| M4 Spatial VLAD | 40960 | 0.1501 | 0.1016 | 0.1851 | 0.1227 | 0.1847 | 6.07 |
| M5 Fisher vector | 16384 | 0.1724 | 0.1136 | 0.2118 | 0.1394 | 0.2089 | 56.0 |
| M6 Fisher + geometric check | 16384 | **0.1968** | **0.1231** | **0.2299** | **0.1502** | **0.2411** | 164.6 |

MAP standard deviation over the 3 seeds is between 0.0002 and 0.0076 for all methods.

Main findings (paired Wilcoxon on per-query AP, see `final_results/wilcoxon.csv`):

- Every method is clearly above random ranking (p < 0.0001).
- M1 TF-IDF beats the baseline M0 (p = 0.0005). Model A and M2 soft assignment are not significantly better than M0.
- The spatial pyramid (B) is slightly but significantly worse than M0 (p = 0.003).
- Fisher + geometric check (M6) is the best method and beats Fisher alone (p < 0.0001), at about 3x the query time.
- Model C (learned) and Fisher are not significantly different (p = 0.92).

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy scipy scikit-learn scikit-image opencv-python h5py pandas matplotlib tqdm joblib torch
python main.py
```


The first run extracts keypoints and caches them in `cache/` (a few minutes); later runs reuse the cache. The cache file name contains the descriptor, keypoint count and image size, so changing those settings never reuses an old cache. A full run with 3 seeds takes a while on CPU (5-fold CNN training, Fisher vectors and geometric verification are repeated for every seed).

## Output

`main.py` writes to `results/` (ignored by git):

- `unified_results.csv`: final table, mean over seeds, with `MAP_std`
- `results_per_seed.csv`: every seed separately
- `wilcoxon.csv`: paired Wilcoxon tests (all methods vs M0 and vs Random, best two methods, Model C vs Fisher)
- `per_query_ap.npz`: per-query AP (mean over seeds)
- `results.png`: bar chart of MAP

The results of the final run are copied to `final_results/`.

## Settings (`src/config.py`)

| Key | Value | Meaning |
|---|---|---|
| `USE_LITE` | True | Lite set (1000 images, 500 queries); False = full set |
| `DESCRIPTOR` | sift | RootSIFT |
| `N_KEYPOINTS` | 2000 | Maximum keypoints per image |
| `MAX_SIDE` | 640 | Images are resized to this maximum side |
| `PCA_DIM` | 64 | Dimension of the local descriptors after PCA |
| `BOW_K`, `K_WORDS` | 128 | Shared visual vocabulary size |
| `VLAD_K`, `FISHER_K` | 128 | Clusters / mixture components |
| `SOFT_N` | 3 | Words per descriptor in soft assignment |
| `GRID_SIZE` | (12, 12) | Spatial grid for models A, B, C |
| `GRID_LEVELS` | (1, 2) | Spatial VLAD grids |
| `GV_SHORTLIST` | 30 | Results re-ranked by geometric verification |
| `N_FOLDS` | 5 | Cross-validation folds for Model C |
| `SEEDS` | (0, 1, 2) | Seeds for PCA, k-means, GMM, folds, CNN |

## Code structure

- `main.py`: runs all models for every seed, evaluates them and writes the results
- `src/config.py`: all settings and paths
- `src/dataset.py`: data loading, triplet dataset for Model C
- `src/extractor.py`: feature extraction, BoW, VLAD, Fisher, spatial grids
- `src/models.py`: Model C network and the spatial pyramid
- `src/evaluate.py`: shared evaluation (MAP, P/R/F1@k, nDCG@k, per-query AP)
- `src/rerank.py`: geometric verification (RANSAC)
- `histogram_replacement.py`: earlier standalone script; its numbers use different settings (500 keypoints, K = 256) and are **not** comparable with the table above

## Limitations

- Results are for the lite set. The full set would give the final numbers but takes much longer.
- VLAD and Fisher vectors are much larger than the BoW histogram (8192 and 16384 against 128 dimensions), so part of their gain may come from the larger vector.
- Model C is trained with `fov >= 0.20` as positives but evaluated on `sim`. The cross-validation is over queries; the database images are shared between the training and test folds.
- Hyperparameters (K, PCA dimension, grid size, CNN settings) are not tuned.
- Geometric verification is slow compared with the other methods.
- `ms_per_query` is measured on a laptop CPU and only meant for comparing methods with each other.
