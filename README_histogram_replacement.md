# Histogram replacement for Bag-of-Visual-Words retrieval

Script: `histogram_replacement.py` (author: Kishor)

Bag of Visual Words (BoVW) turns an image into a histogram that only counts visual words. This script keeps everything before the histogram (keypoints, local descriptors, visual vocabulary) and replaces only the "descriptors to one vector per image" step. All methods use the same descriptors, queries and metrics.

## Methods

| ID | Method | Idea |
|---|---|---|
| M0 | BoW baseline | Hard-assignment counts, z-score, Euclidean distance (Assignment 02 style) |
| M1 | BoW TF-IDF | Counts, log scaling, IDF, L2 normalisation, cosine |
| M2 | Soft-assignment BoW | Each descriptor votes for its 3 nearest words |
| M3 | VLAD | Sum of residuals (descriptor minus centroid) per word, with intra-, power- and L2-normalisation |
| M4 | Spatial VLAD | VLAD per cell of a 1x1 and 2x2 grid, concatenated (keeps where features are) |
| M5 | Fisher vector | Diagonal GMM, gradients with respect to mean and variance |
| M6 | Best method + geometric verification | Re-ranks the top 30 results by RANSAC inliers between matched keypoints |

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy scipy scikit-learn opencv-python h5py pandas matplotlib tqdm
python histogram_replacement.py
```

The dataset folder (`data02`, from FIR-02.zip) must sit next to the script, run the script from the folder that contains it. The first run extracts keypoints and caches them in `cache/` (about 1 to 3 minutes on the lite set), later runs reuse the cache.

## Output (folder `results/`)

- `results.csv`: one row per method (MAP, P@5/10, R@5/10, F1@5/10, vector size, ms per query)
- `wilcoxon.csv`: paired Wilcoxon tests on per-query AP against M0 and M1
- `per_query_ap.npz`: per-query AP for every method
- `results.png`: bar chart of MAP

## Settings (`CONFIG` at the top of the script)

| Key | Meaning |
|---|---|
| `data_dir` | Folder with `database/`, `query/` and the `.h5` ground truth |
| `use_full` | `False` = lite set (1000 images, 500 queries), `True` = full set (3291 images, 2692 queries) |
| `descriptor` | `"sift"` (RootSIFT) or `"orb"` (ORB mode not yet tested) |
| `n_keypoints` | Keypoints per image |
| `pca_dim` | Local descriptors are reduced to this size |
| `bow_k`, `vlad_k`, `fisher_k` | Vocabulary sizes of the histogram, VLAD and Fisher methods |
| `soft_n` | Number of words a descriptor votes for in soft assignment |
| `grid_levels` | Spatial VLAD grids |
| `gv_shortlist` | How many top results geometric verification re-ranks |
| `seed` | Random seed for PCA, k-means and GMM |


## Using this with the group's code

The shared contract: **every method produces a similarity matrix `S` with shape `[n_queries, n_database]`** (higher = more similar), with rows and columns in the same order as the `query*.json` and `database*.json` lists. Then:

```python
import histogram_replacement as hr

map_paths, q_paths, gt, tag = hr.load_data()
assert S.shape == gt.shape
row, per_query_ap, order = hr.evaluate(S, gt, "Jesse BoVW", enc_time=0.0, dim=K)
```

`row` holds MAP, P@k, R@k and F1@k, and `per_query_ap` is what the Wilcoxon tests need (every method must be evaluated on the same queries). Queries without any relevant image are skipped (46 of 2692 in the full set), so everyone gets the same MAP definition.

**Jesse (tuned BoVW baseline)**
1. Turn the best configuration from the sweep into `S` (for Euclidean distance use `S = -distance_matrix`).
2. Evaluate with `hr.evaluate` as above, so the baseline uses the same metric code as everything else.
3. Report the mean and standard deviation over several seeds, not the single best run.
4. Push the sweep script (only the notebook and the results CSV are in the branch now).

**Nathan (spatial pyramid and CNN)**
1. Turn the cosine similarities (or the CNN embeddings, `Q @ M.T`) into `S` and evaluate with `hr.evaluate`. Keep nDCG@k on `fov` from `src/evaluate.py` as an extra metric.
2. Train the CNN only on queries that are not used for evaluation. Evaluate every method on the same held-out queries when comparing with the CNN, for example `hr.evaluate(S[test_idx], gt[test_idx], ...)`.
3. Make the image loader fail loudly on unreadable images, because skipping one shifts all later indices against the ground truth.
4. Use `data_dir="data"` (his folder name) or rename the folder, the internal layout is the same.



**Before committing:** the repository `.gitignore` does not exclude `data/`, `cache/` or `results/`. Add these lines so images and caches are not uploaded:

```
data/
data02/
cache/
results/
*.pkl
*.npy
*.joblib
*.h5
```

## Limitations

- Results are for the lite set and one seed. Run `use_full=True` for the final numbers (geometric verification takes about 10 minutes there).
- M0 here uses SIFT with 500 keypoints and k=256, so it is not identical to the Assignment 02 baseline (ORB, 50 keypoints, k=32).
- VLAD and Fisher vectors are larger than the histogram, so part of their gain may come from more dimensions.
- Hyperparameters (`vlad_k`, `fisher_k`, `pca_dim`) are untuned first guesses.
- Geometric verification is slow compared with the other methods (about 31 ms per query on the lite set against under 5 ms).
