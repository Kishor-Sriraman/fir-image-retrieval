import os
import joblib
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans

class SpatialVisualWordExtractor:
    """Extracts ORB features, fits visual vocabulary, and quantizes images into spatial grids."""
    def __init__(self, n_clusters=128, grid_size=(12, 12)):
        self.n_clusters = n_clusters
        self.grid_size = grid_size  # (gh_size, gw_size)
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.kmeans = None
        self.idf = None

    def fit_vocabulary(self, image_list, cache_path=None):
        """Fits or loads K-Means visual vocabulary."""
        if cache_path and os.path.exists(cache_path):
            print(f"[EXTRACTOR] Loading cached vocabulary from {cache_path}...")
            self.kmeans = joblib.load(cache_path)
            return self.kmeans

        print(f"[EXTRACTOR] Fitting K-Means (K={self.n_clusters})...")
        descriptors = []
        for img in image_list:
            _, des = self.orb.detectAndCompute(_to_gray(img), None)
            if des is not None:
                descriptors.append(des)

        all_descriptors = np.vstack(descriptors).astype(np.float32)
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=2048)
        self.kmeans.fit(all_descriptors)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            joblib.dump(self.kmeans, cache_path)

        return self.kmeans

    def process_image(self, img):
        """Computes global histogram and spatial grid representation for an image."""
        gray = _to_gray(img)
        h, w = gray.shape
        kp, des = self.orb.detectAndCompute(gray, None)

        gh_size, gw_size = self.grid_size
        spatial_grid = np.zeros((self.n_clusters, gh_size, gw_size), dtype=np.float32)
        global_hist = np.zeros(self.n_clusters, dtype=np.float32)

        if des is None or not kp:
            return global_hist, spatial_grid

        words = self.kmeans.predict(des.astype(np.float32))

        for keypoint, word in zip(kp, words):
            x, y = keypoint.pt
            gx = min(gw_size - 1, max(0, int((x / w) * gw_size)))
            gy = min(gh_size - 1, max(0, int((y / h) * gh_size)))

            global_hist[word] += 1.0 # For the normal BoVW
            spatial_grid[word, gy, gx] += 1.0 # For the spatial grid BoVW

        return global_hist, spatial_grid

    def compute_idf(self, spatial_grids):
        """Computes Inverse Document Frequency (IDF) weights."""
        num_docs = spatial_grids.shape[0]
        doc_freq = (spatial_grids > 0).sum(axis=(0, 2, 3), keepdims=True) + 1e-5
        self.idf = np.log((num_docs + 1.0) / doc_freq)
        return self.idf

    def apply_tfidf(self, spatial_grids):
        """Applies L1 term frequency normalization and IDF scaling."""
        tf = spatial_grids / (spatial_grids.sum(axis=1, keepdims=True) + 1e-8)
        return tf * self.idf if self.idf is not None else tf


def _to_gray(img):
    """Ensures input image is uint8 grayscale."""
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

def compute_features(extractor, image_list, hist_cache, grid_cache, desc="Images"):
    """Loads or computes global histograms and spatial grids."""
    if os.path.exists(hist_cache) and os.path.exists(grid_cache):
        print(f"[EXTRACTOR] Loading cached features for {desc}...")
        return np.load(hist_cache), np.load(grid_cache)

    print(f"[EXTRACTOR] Extracting features for {desc}...")
    hists, grids = zip(*[extractor.process_image(img) for img in image_list])
    
    hists_arr, grids_arr = np.array(hists, dtype=np.float32), np.array(grids, dtype=np.float32)

    os.makedirs(os.path.dirname(hist_cache), exist_ok=True)
    np.save(hist_cache, hists_arr)
    np.save(grid_cache, grids_arr)

    return hists_arr, grids_arr