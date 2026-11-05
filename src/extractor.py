import os
import joblib
import numpy as np
import cv2
import pickle
from skimage import color
from sklearn.cluster import KMeans
import src.config as config


def _to_gray(img):
    """Ensures input image is 2D float/uint8 grayscale."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = color.rgba2rgb(img)
        return color.rgb2gray(img)
    return img


class SpatialVisualWordExtractor:
    """Extracts local features, fits visual vocabulary, and maps visual words to spatial grids."""
    def __init__(self, n_clusters=64, grid_size=(16, 16), n_keypoints=500):
        self.n_clusters = n_clusters
        self.grid_size = grid_size
        
        # Initialize detector based on config setting
        if config.DESCRIPTOR.lower() == "sift":
            self.detector = cv2.SIFT_create(nfeatures=n_keypoints)
        else:
            self.detector = cv2.ORB_create(nfeatures=n_keypoints)
            
        self.kmeans = None
        self.idf = None

    def fit_vocabulary(self, images, cache_path=None):
        """Fits or loads K-Means visual vocabulary from pre-loaded image arrays."""
        if cache_path and os.path.exists(cache_path):
            print(f"[CACHE] Loading cached vocabulary from {cache_path}...")
            self.kmeans = joblib.load(cache_path)
            return self.kmeans

        all_descriptors = []
        for img in images:
            img_gray = _to_gray(img)
            if img_gray.dtype != np.uint8:
                img_gray = (img_gray * 255).astype(np.uint8) if img_gray.max() <= 1.0 else img_gray.astype(np.uint8)
                
            _, des = self.detector.detectAndCompute(img_gray, None)
            if des is not None and len(des) > 0:
                all_descriptors.append(des.astype(np.float32))

        if not all_descriptors:
            raise ValueError("[ERROR] No descriptors found across any images in the dataset! Check image paths or detector settings.")

        all_descriptors = np.vstack(all_descriptors)
        print(f"Fitting K-Means vocabulary on {all_descriptors.shape[0]} local descriptors...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=config.SEED, n_init=10)
        self.kmeans.fit(all_descriptors)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            joblib.dump(self.kmeans, cache_path)
            print(f"[CACHE] Saved K-Means vocabulary to {cache_path}")

        return self.kmeans

    def image_to_features(self, img):
        """Extracts keypoints and returns both global BoVW histogram and spatial visual word map."""
        img_gray = _to_gray(img)
        if img_gray.dtype != np.uint8:
            img_gray = (img_gray * 255).astype(np.uint8) if img_gray.max() <= 1.0 else img_gray.astype(np.uint8)
            
        h_img, w_img = img_gray.shape
        kp, descs = self.detector.detectAndCompute(img_gray, None)
        
        grid_h, grid_w = self.grid_size
        global_hist = np.zeros(self.n_clusters, dtype=np.float32)
        spatial_map = np.zeros((self.n_clusters, grid_h, grid_w), dtype=np.float32)

        if descs is None or len(descs) == 0 or not kp:
            return global_hist, spatial_map

        word_indices = self.kmeans.predict(descs.astype(np.float32))

        for keypoint, word_id in zip(kp, word_indices):
            x, y = keypoint.pt
            global_hist[word_id] += 1.0

            # Bilinear interpolation for soft-assignment coordinates
            r_float = (y / h_img) * (grid_h - 1)
            c_float = (x / w_img) * (grid_w - 1)

            r0, c0 = int(np.floor(r_float)), int(np.floor(c_float))
            r1, c1 = min(r0 + 1, grid_h - 1), min(c0 + 1, grid_w - 1)
            dr, dc = r_float - r0, c_float - c0

            spatial_map[word_id, r0, c0] += (1 - dr) * (1 - dc)
            spatial_map[word_id, r0, c1] += (1 - dr) * dc
            spatial_map[word_id, r1, c0] += dr * (1 - dc)
            spatial_map[word_id, r1, c1] += dr * dc

        return global_hist, spatial_map

    def compute_idf(self, database_grids):
        """Calculates Inverse Document Frequency (IDF) over spatial grid channels."""
        N = database_grids.shape[0]
        word_presence = (database_grids.sum(axis=(2, 3)) > 0).astype(np.float32)
        doc_freq = word_presence.sum(axis=0) + 1e-5
        self.idf = np.log((N + 1.0) / doc_freq)[None, :, None, None]

    def apply_tfidf(self, grids):
        """Applies L1 term-frequency normalization and IDF weighting."""
        tf = grids / (grids.sum(axis=1, keepdims=True) + 1e-8)
        return tf * self.idf if hasattr(self, 'idf') and self.idf is not None else tf


def compute_features(extractor, image_list, hist_cache, grid_cache, desc="Images"):
    """Loads or computes global histograms and spatial grids for your models."""
    if os.path.exists(hist_cache) and os.path.exists(grid_cache):
        print(f"[EXTRACTOR] Loading cached features for {desc}...")
        return np.load(hist_cache), np.load(grid_cache)

    print(f"[EXTRACTOR] Extracting features for {desc}...")
    hists, grids = zip(*[extractor.image_to_features(img) for img in image_list])
    
    hists_arr = np.array(hists, dtype=np.float32)
    grids_arr = np.array(grids, dtype=np.float32)

    os.makedirs(os.path.dirname(hist_cache), exist_ok=True)
    np.save(hist_cache, hists_arr)
    np.save(grid_cache, grids_arr)

    return hists_arr, grids_arr

# Kishor's code

def extract_all_from_images(images, cache_path=None):
    """
    Extracts raw features directly from pre-loaded image arrays using config settings,
    returning structured tuples `(descriptors, keypoints_xy, shape)` for partner models.
    """
    # Ensure cache path ends with .pkl since we are pickling lists of variable-length arrays
    if cache_path and not cache_path.endswith('.pkl'):
        cache_path = os.path.splitext(cache_path)[0] + '.pkl'

    if cache_path and os.path.exists(cache_path):
        print(f"[EXTRACTOR] Loading cached raw partner features from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[EXTRACTOR] Extracting raw partner features for {len(images)} images using {config.DESCRIPTOR.upper()}...")
    
    if config.DESCRIPTOR.lower() == "sift":
        detector = cv2.SIFT_create(nfeatures=config.N_KEYPOINTS)
    else:
        detector = cv2.ORB_create(nfeatures=config.N_KEYPOINTS)

    raw_features = []
    for img in images:
        h, w = img.shape[:2]
        if max(h, w) > config.MAX_SIDE:
            scale = config.MAX_SIDE / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        img_gray = _to_gray(img)
        if img_gray.dtype != np.uint8:
            img_gray = (img_gray * 255).astype(np.uint8) if img_gray.max() <= 1.0 else img_gray.astype(np.uint8)

        kp, des = detector.detectAndCompute(img_gray, None)
        
        xy = np.array([k.pt for k in kp], dtype=np.float32) / np.array([w, h], dtype=np.float32) if kp else np.zeros((0, 2), dtype=np.float32)
        
        if des is not None:
            desc = des.astype(np.float32)
            if config.DESCRIPTOR.lower() == "sift":
                desc /= desc.sum(axis=1, keepdims=True) + 1e-7
                desc = np.sqrt(desc)
        else:
            desc = np.zeros((0, 128 if config.DESCRIPTOR.lower() == "sift" else 32), dtype=np.float32)
            
        raw_features.append((desc, xy, (w, h)))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(raw_features, f)
        print(f"[EXTRACTOR] Saved raw partner features to {cache_path}")

    return raw_features


def stack(feats, max_n=200_000, seed=None):
    """Stacks all local descriptors into a single array for PCA/K-Means fitting."""
    all_desc = [d[0] for d in feats if len(d[0]) > 0]
    if not all_desc:
        return np.zeros((0, config.PCA_DIM), dtype=np.float32)
    arr = np.vstack(all_desc)
    if len(arr) > max_n:
        idx = np.random.RandomState(config.SEED if seed is None else seed).choice(len(arr), max_n, replace=False)
        arr = arr[idx]
    return arr.astype(np.float32)


def l2n(x, axis=1):
    """L2 Normalization helper."""
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def encode_set(feats, fn, dim):
    """Applies a given encoding function over a set of feature tuples."""
    out = np.zeros((len(feats), dim), np.float32)
    for i, (d, xy, _) in enumerate(feats):
        if len(d):
            out[i] = fn(d, xy)
    return out


def bow_counts(feats, centers, soft_n=1):
    """Computes BoW or soft-assignment histograms."""
    K = len(centers)
    c2 = (centers ** 2).sum(axis=1)

    def fn(d, xy):
        dist = (d ** 2).sum(axis=1, keepdims=True) - 2 * d @ centers.T + c2
        if soft_n == 1:
            return np.bincount(dist.argmin(axis=1), minlength=K).astype(np.float32)
        idx = np.argpartition(dist, soft_n, axis=1)[:, :soft_n]
        dd = np.take_along_axis(dist, idx, 1)
        sigma2 = np.median(dd) + 1e-9
        w = np.exp(-dd / (2 * sigma2))
        w /= w.sum(axis=1, keepdims=True)
        h = np.zeros(K, dtype=np.float32)
        np.add.at(h, idx.ravel(), w.ravel())
        return h

    return encode_set(feats, fn, K)


def tfidf_l2(Hm, Hq):
    """Applies TF-IDF weighting and L2 normalization."""
    df = (Hm > 0).sum(axis=0)
    idf = np.log((1 + len(Hm)) / (1 + df)) + 1
    f = lambda H: l2n(np.log1p(H) * idf)
    return f(Hm), f(Hq)


def vlad_vec(d, centers, K, D):
    """Computes a single VLAD vector for an image."""
    c2 = (centers ** 2).sum(axis=1)
    a = ((d ** 2).sum(axis=1, keepdims=True) - 2 * d @ centers.T + c2).argmin(axis=1)
    v = np.zeros((K, D), dtype=np.float32)
    np.add.at(v, a, d - centers[a])
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    v = np.sign(v) * np.sqrt(np.abs(v))
    return v.ravel()


def vlad_encode(feats, centers, spatial_levels=None):
    """Computes standard or spatial VLAD representations."""
    K, D = centers.shape
    if spatial_levels is None:
        return l2n(encode_set(feats, lambda d, xy: vlad_vec(d, centers, K, D), K * D))
    
    cells = sum(g * g for g in spatial_levels)

    def fn(d, xy):
        parts = []
        for g in spatial_levels:
            ix = np.minimum((xy[:, 0] * g).astype(int), g - 1)
            iy = np.minimum((xy[:, 1] * g).astype(int), g - 1)
            cell = iy * g + ix
            for c in range(g * g):
                m = cell == c
                parts.append(vlad_vec(d[m], centers, K, D) if m.any() else np.zeros(K * D, dtype=np.float32))
        return np.concatenate(parts) / np.sqrt(cells)

    return l2n(encode_set(feats, fn, K * D * cells))


def fisher_encode(feats, gmm):
    """Computes Fisher Vector representations using GMM parameters."""
    K, D = gmm.means_.shape
    w = gmm.weights_.astype(np.float32)
    mu = gmm.means_.astype(np.float32)
    var = gmm.covariances_.astype(np.float32)
    sig = np.sqrt(var)

    def fn(d, xy):
        g = gmm.predict_proba(d).astype(np.float32)
        n = len(d)
        G_mu = np.zeros((K, D), dtype=np.float32)
        G_sig = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            diff = (d - mu[k]) / sig[k]
            G_mu[k] = (g[:, k:k + 1] * diff).sum(axis=0) / (n * np.sqrt(w[k]))
            G_sig[k] = (g[:, k:k + 1] * (diff ** 2 - 1)).sum(axis=0) / (n * np.sqrt(2 * w[k]))
        v = np.concatenate([G_mu.ravel(), G_sig.ravel()])
        v = np.sign(v) * np.sqrt(np.abs(v))
        return v

    return l2n(encode_set(feats, fn, 2 * K * D))


# ---------------------------------------------------------------------------
# Spatial visual-word grids built from the SAME (PCA-projected) local features
# and the SAME vocabulary as the BoW / VLAD / Fisher models.
# ---------------------------------------------------------------------------
def spatial_word_grids(feats, centers, grid_size=(12, 12)):
    """Returns (N, K, H, W) grids: each keypoint votes for its nearest word at its (bilinearly
    interpolated) position. grids.sum(axis=(2, 3)) equals the hard-assignment BoW histogram."""
    K = len(centers)
    gh, gw = grid_size
    c2 = (centers ** 2).sum(axis=1)
    grids = np.zeros((len(feats), K, gh, gw), dtype=np.float32)
    for i, (d, xy, _) in enumerate(feats):
        if len(d) == 0:
            continue
        words = ((d ** 2).sum(axis=1, keepdims=True) - 2 * d @ centers.T + c2).argmin(axis=1)
        xy = np.clip(xy, 0.0, 1.0)
        r, c = xy[:, 1] * (gh - 1), xy[:, 0] * (gw - 1)
        r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
        r1, c1 = np.minimum(r0 + 1, gh - 1), np.minimum(c0 + 1, gw - 1)
        dr, dc = (r - r0).astype(np.float32), (c - c0).astype(np.float32)
        np.add.at(grids[i], (words, r0, c0), (1 - dr) * (1 - dc))
        np.add.at(grids[i], (words, r0, c1), (1 - dr) * dc)
        np.add.at(grids[i], (words, r1, c0), dr * (1 - dc))
        np.add.at(grids[i], (words, r1, c1), dr * dc)
    return grids


def grid_idf(db_grids):
    """IDF per visual word, computed on the database grids only."""
    n = db_grids.shape[0]
    presence = (db_grids.sum(axis=(2, 3)) > 0).astype(np.float32)
    df = presence.sum(axis=0) + 1e-5
    return np.log((n + 1.0) / df)[None, :, None, None].astype(np.float32)


def grid_tfidf(grids, idf):
    """L1 term-frequency normalisation over the word axis, then IDF weighting."""
    tf = grids / (grids.sum(axis=1, keepdims=True) + 1e-8)
    return np.ascontiguousarray(tf * idf, dtype=np.float32)