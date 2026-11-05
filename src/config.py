import os

# -----------------------------------------------------------------------------
# Dataset & Execution Settings
# -----------------------------------------------------------------------------
USE_LITE = True
USE_FULL = not USE_LITE
SEED = 0

# Feature Extraction Settings
DESCRIPTOR = "sift"       # Feature type: "sift" or "orb"
N_KEYPOINTS = 2000         # Max keypoints per image
MAX_SIDE = 640            # Max image dimension for resizing
PCA_DIM = 64              # Dimensionality reduction target for local descriptors

# Model Hyperparameters
# One shared visual vocabulary size for every BoW-based model (A, B, C, M0, M1, M2),
# so the BoVW part of the experiment is a constant.
BOW_K = 128               # BoW visual vocabulary size
K_WORDS = BOW_K           # Spatial models (A, B, C) reuse the same vocabulary
GRID_SIZE = (12, 12)      # Spatial grid resolution (build_spatial_pyramid needs 12x12)

VLAD_K = 128              # VLAD cluster count
FISHER_K = 128            # GMM mixture count for Fisher Vectors
SOFT_N = 3                # Soft-assignment top-N nearest clusters
GRID_LEVELS = (1, 2)      # Spatial levels for Spatial VLAD (e.g., 1x1, 2x2)

# Learned model (Model C) settings
N_FOLDS = 5               # Cross-validation folds over the queries
CNN_EPOCHS = 15
CNN_LR = 1e-3
CNN_EMB_DIM = 256
CNN_BATCH = 32

# Seeds: results are averaged over these (use (0, 1, 2) for the final run)
SEEDS = (0,1,2)

# Geometric Verification (GV / RANSAC) Settings
GV_SHORTLIST = 30         # Number of top candidate images to rerank
GV_RATIO = 0.8            # Lowe's ratio test threshold for matching
GV_MIN_INLIERS = 4        # Minimum RANSAC inliers required for valid match

# Evaluation Settings
K_LIST = (5, 10)          # Evaluation cut-offs for Precision/F1@K and nDCG@K

# -----------------------------------------------------------------------------
# Path Definitions
# -----------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

MODE_STR = "_lite" if USE_LITE else ""

# Directory containing images
QUERY_IMG_DIR = os.path.join(DATA_DIR, "query", "images")
DATABASE_IMG_DIR = os.path.join(DATA_DIR, "database", "images")

# JSON list paths
QUERY_JSON = os.path.join(DATA_DIR, "query", f"query{MODE_STR}.json")
DATABASE_JSON = os.path.join(DATA_DIR, "database", f"database{MODE_STR}.json")

# Ground Truth H5 path
GT_FILE = os.path.join(DATA_DIR, f"london{MODE_STR}_gt.h5")

# Cache Files
KMEANS_CACHE = os.path.join(CACHE_DIR, f"kmeans_vocab_{'lite' if USE_LITE else 'full'}.joblib")