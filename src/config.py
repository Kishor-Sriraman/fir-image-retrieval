import os

# Change these hyperparameters
USE_LITE = True
K_WORDS = 128
GRID_SIZE = (12, 12)

# Path definitions
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


MODE_STR = "_lite" if USE_LITE else ""

# Directory containing images
QUERY_IMG_DIR = os.path.join(DATA_DIR, "query", "images")
DATABASE_IMG_DIR = os.path.join(DATA_DIR, "database", "images")

# JSON list paths
QUERY_JSON = os.path.join(DATA_DIR, "query", f"query{MODE_STR}.json")
DATABASE_JSON = os.path.join(DATA_DIR, "database", f"database{MODE_STR}.json")

# Ground Truth H5 path
GT_FILE = os.path.join(DATA_DIR, f"london{MODE_STR}_gt.h5")

KMEANS_CACHE = os.path.join(CACHE_DIR, f"kmeans_vocab{'lite' if USE_LITE else 'full'}.joblib")