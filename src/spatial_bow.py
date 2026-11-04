import os
import joblib
import numpy as np
from skimage import color
from skimage.feature import ORB
from sklearn.cluster import KMeans


def _to_gray(img):
    """Ensures input image is 2D float/uint8 grayscale."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = color.rgba2rgb(img)
        return color.rgb2gray(img)
    return img


class SpatialVisualWordExtractor:
    """Extracts ORB features and maps visual words to spatial grids using bilinear soft-assignment.
        Output: 1D array of global histogram and 3D tensor spatial grid.
    """
    def __init__(self, n_clusters=64, grid_size=(16, 16), n_keypoints=500):
        self.n_clusters = n_clusters
        self.grid_size = grid_size
        self.extractor = ORB(n_keypoints=n_keypoints)
        self.kmeans = None

    def fit_vocabulary(self, images, cache_path=None):
        """Fits / load K-Means visual vocabulary."""
        if cache_path and os.path.exists(cache_path):
            print(f"[CACHE] Loading cached vocabulary from {cache_path}...")
            self.kmeans = joblib.load(cache_path)
            return

        # retreive local ORB features and append to one big list per image
        all_descriptors = []
        for img in images:
            self.extractor.detect_and_extract(_to_gray(img))
            if self.extractor.descriptors is not None and len(self.extractor.descriptors) > 0:
                all_descriptors.append(self.extractor.descriptors)

        # Cluster all descriptors int K visual words
        all_descriptors = np.vstack(all_descriptors)
        print(f"Fitting K-Means vocabulary on {all_descriptors.shape[0]} local descriptors...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(all_descriptors)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            joblib.dump(self.kmeans, cache_path)
            print(f"[CACHE] Saved K-Means vocabulary to {cache_path}")

    def image_to_features(self, img):
        """Extracts keypoints once and returns both global BoVW histogram and spatial visual word map."""
        img_gray = _to_gray(img)
        h_img, w_img = img_gray.shape
        self.extractor.detect_and_extract(img_gray)
        
        kps = self.extractor.keypoints
        descs = self.extractor.descriptors
        grid_h, grid_w = self.grid_size

        global_hist = np.zeros(self.n_clusters, dtype=np.float32)
        spatial_map = np.zeros((self.n_clusters, grid_h, grid_w), dtype=np.float32)

        if descs is None or len(descs) == 0:
            return global_hist, spatial_map

        # map each local  descriptor to its closest visual  word
        word_indices = self.kmeans.predict(descs)

        for (y, x), word_id in zip(kps, word_indices):
            # Update global histogram
            global_hist[word_id] += 1.0

            # Bilinear interpolation for soft-assignment coordinates
            r_float = (y / h_img) * (grid_h - 1)
            c_float = (x / w_img) * (grid_w - 1)

            # Top-left grid indices and fractional offsets
            r0, c0 = int(np.floor(r_float)), int(np.floor(c_float))
            r1, c1 = min(r0 + 1, grid_h - 1), min(c0 + 1, grid_w - 1)
            dr, dc = r_float - r0, c_float - c0

            # distribute keypoints softly across 4 neighbour grid cells
            spatial_map[word_id, r0, c0] += (1 - dr) * (1 - dc)
            spatial_map[word_id, r0, c1] += (1 - dr) * dc
            spatial_map[word_id, r1, c0] += dr * (1 - dc)
            spatial_map[word_id, r1, c1] += dr * dc

        return global_hist, spatial_map

    def compute_idf(self, database_grids):
        """Calculates Inverse Document Frequency (IDF) over spatial grid channels."""
        N = database_grids.shape[0]
        # Check if a visual word channel is present anywhere in each database grid
        word_presence = (database_grids.sum(axis=(2, 3)) > 0).astype(np.float32)
        doc_freq = word_presence.sum(axis=0) + 1e-5
        self.idf = np.log((N + 1.0) / doc_freq)[None, :, None, None]

    def apply_tfidf(self, grids):
        """Applies L1 term-frequency normalization and IDF weighting."""
        tf = grids / (grids.sum(axis=1, keepdims=True) + 1e-8)
        return tf * self.idf if hasattr(self, 'idf') else tf