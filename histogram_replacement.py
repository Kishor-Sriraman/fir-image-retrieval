import os, json, time, pickle, warnings
import numpy as np, pandas as pd, h5py, cv2
from scipy.stats import wilcoxon
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

warnings.filterwarnings("ignore")


CONFIG = dict(
    data_dir="data02",
    use_full=False,
    descriptor="sift",
    n_keypoints=500,
    max_side=640,
    pca_dim=64,
    bow_k=256,
    vlad_k=32,
    fisher_k=16,
    soft_n=3,
    grid_levels=(1, 2),
    gv_shortlist=30,
    gv_ratio=0.8,
    gv_min_inliers=4,
    k_list=(5, 10),
    cache_dir="cache",
    out_dir="results",
    seed=0,
)
C = CONFIG
rng = np.random.default_rng(C["seed"])
os.makedirs(C["cache_dir"], exist_ok=True)
os.makedirs(C["out_dir"], exist_ok=True)


def load_data():
    d = C["data_dir"]
    if C["use_full"]:
        db_json, q_json, gt_file, tag = "database/database.json", "query/query.json", "london_gt.h5", "full"
    else:
        db_json, q_json, gt_file, tag = "database/database_lite.json", "query/query_lite.json", "london_lite_gt.h5", "lite"
    map_paths = json.load(open(os.path.join(d, db_json)))["im_paths"]
    q_paths = json.load(open(os.path.join(d, q_json)))["im_paths"]
    sim = h5py.File(os.path.join(d, gt_file), "r")["sim"][:].astype(bool)
    assert sim.shape == (len(q_paths), len(map_paths)), (sim.shape, len(q_paths), len(map_paths))
    return map_paths, q_paths, sim, tag


def make_detector():
    if C["descriptor"] == "sift":
        return cv2.SIFT_create(nfeatures=C["n_keypoints"])
    return cv2.ORB_create(nfeatures=C["n_keypoints"])


def extract_one(path, det):

    img = cv2.imread(os.path.join(C["data_dir"], path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((0, 128), np.float32), np.zeros((0, 2), np.float32), (1, 1)
    h, w = img.shape
    s = C["max_side"] / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape
    kps, desc = det.detectAndCompute(img, None)
    if desc is None or len(kps) == 0:
        return np.zeros((0, 128 if C["descriptor"] == "sift" else 256), np.float32), np.zeros((0, 2), np.float32), (w, h)
    xy = np.array([k.pt for k in kps], np.float32) / np.array([w, h], np.float32)
    if C["descriptor"] == "sift":
        desc = desc.astype(np.float32)
        desc /= desc.sum(1, keepdims=True) + 1e-7
        desc = np.sqrt(desc)
    else:
        desc = np.unpackbits(desc, axis=1).astype(np.float32)
    return desc, xy, (w, h)


def extract_all(paths, name):
    f = os.path.join(C["cache_dir"], f"{name}_{C['descriptor']}_{C['n_keypoints']}_{C['max_side']}.pkl")
    if os.path.exists(f):
        print(f"[cache] loaded {f}")
        return pickle.load(open(f, "rb"))
    det = make_detector()
    out = [extract_one(p, det) for p in tqdm(paths, desc=f"extract {name}")]
    pickle.dump(out, open(f, "wb"))
    return out


def stack(feats, max_n=None):
    d = np.vstack([f[0] for f in feats if len(f[0])])
    if max_n and len(d) > max_n:
        d = d[rng.choice(len(d), max_n, replace=False)]
    return d


def l2n(x, axis=1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def encode_set(feats, fn, dim):
    out = np.zeros((len(feats), dim), np.float32)
    for i, (d, xy, _) in enumerate(feats):
        if len(d):
            out[i] = fn(d, xy)
    return out


def bow_counts(feats, centers, soft_n=1):
    K = len(centers)
    c2 = (centers ** 2).sum(1)

    def fn(d, xy):
        dist = (d ** 2).sum(1, keepdims=True) - 2 * d @ centers.T + c2
        if soft_n == 1:
            return np.bincount(dist.argmin(1), minlength=K).astype(np.float32)
        idx = np.argpartition(dist, soft_n, axis=1)[:, :soft_n]
        dd = np.take_along_axis(dist, idx, 1)
        sigma2 = np.median(dd) + 1e-9
        w = np.exp(-dd / (2 * sigma2))
        w /= w.sum(1, keepdims=True)
        h = np.zeros(K, np.float32)
        np.add.at(h, idx.ravel(), w.ravel())
        return h

    return encode_set(feats, fn, K)


def tfidf_l2(Hm, Hq):
    df = (Hm > 0).sum(0)
    idf = np.log((1 + len(Hm)) / (1 + df)) + 1
    f = lambda H: l2n(np.log1p(H) * idf)
    return f(Hm), f(Hq)


def vlad_vec(d, centers, K, D):
    c2 = (centers ** 2).sum(1)
    a = ((d ** 2).sum(1, keepdims=True) - 2 * d @ centers.T + c2).argmin(1)
    v = np.zeros((K, D), np.float32)
    np.add.at(v, a, d - centers[a])
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    v = np.sign(v) * np.sqrt(np.abs(v))
    return v.ravel()


def vlad_encode(feats, centers, spatial_levels=None):
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
                parts.append(vlad_vec(d[m], centers, K, D) if m.any() else np.zeros(K * D, np.float32))
        return np.concatenate(parts) / np.sqrt(cells)

    return l2n(encode_set(feats, fn, K * D * cells))


def fisher_encode(feats, gmm):
    K, D = gmm.means_.shape
    w, mu, var = gmm.weights_.astype(np.float32), gmm.means_.astype(np.float32), gmm.covariances_.astype(np.float32)
    sig = np.sqrt(var)

    def fn(d, xy):
        g = gmm.predict_proba(d).astype(np.float32)
        n = len(d)
        G_mu, G_sig = np.zeros((K, D), np.float32), np.zeros((K, D), np.float32)
        for k in range(K):
            diff = (d - mu[k]) / sig[k]
            G_mu[k] = (g[:, k:k + 1] * diff).sum(0) / (n * np.sqrt(w[k]))
            G_sig[k] = (g[:, k:k + 1] * (diff ** 2 - 1)).sum(0) / (n * np.sqrt(2 * w[k]))
        v = np.concatenate([G_mu.ravel(), G_sig.ravel()])
        v = np.sign(v) * np.sqrt(np.abs(v))
        return v

    return l2n(encode_set(feats, fn, 2 * K * D))


def evaluate(S, gt, name, enc_time, dim):

    t0 = time.time()
    order = np.argsort(-S, axis=1)
    search_t = time.time() - t0
    ap, res = [], {f"{m}@{k}": [] for k in C["k_list"] for m in ("P", "R", "F1")}
    valid = np.where(gt.sum(1) > 0)[0]
    for q in valid:
        rel = gt[q][order[q]]
        n_rel = rel.sum()
        cum = np.cumsum(rel)
        ap.append((cum[rel] / (np.where(rel)[0] + 1)).sum() / n_rel)
        for k in C["k_list"]:
            p, r = cum[k - 1] / k, cum[k - 1] / n_rel
            res[f"P@{k}"].append(p); res[f"R@{k}"].append(r)
            res[f"F1@{k}"].append(0 if p + r == 0 else 2 * p * r / (p + r))
    row = dict(method=name, dim=dim, MAP=np.mean(ap), **{k: np.mean(v) for k, v in res.items()},
               ms_per_query=1000 * (enc_time + search_t) / len(S))
    return row, np.array(ap), order


def geometric_rerank(order, S, q_feats, m_feats, N):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    new_order = order.copy()
    for q in tqdm(range(len(order)), desc="geometric verification"):
        qd, qxy, (qw, qh) = q_feats[q]
        short = order[q, :N]
        inl = np.zeros(N)
        if len(qd) >= C["gv_min_inliers"]:
            for j, m in enumerate(short):
                md, mxy, (mw, mh) = m_feats[m]
                if len(md) < 2:
                    continue
                knn = bf.knnMatch(qd, md, k=2)
                good = [a for a, b in (x for x in knn if len(x) == 2) if a.distance < C["gv_ratio"] * b.distance]
                if len(good) < C["gv_min_inliers"]:
                    continue
                p1 = qxy[[g.queryIdx for g in good]] * [qw, qh]
                p2 = mxy[[g.trainIdx for g in good]] * [mw, mh]
                _, mask = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=0.03 * max(mw, mh))
                inl[j] = 0 if mask is None else int(mask.sum())

        sh_order = np.lexsort((np.arange(N), -inl))
        new_order[q, :N] = short[sh_order]
    return new_order


def eval_order(order, gt, name, dim, t):
    S_fake = np.zeros(order.shape)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(order.shape[1])[None, :].repeat(len(order), 0), 1)
    S_fake = -rank.astype(float)
    return evaluate(S_fake, gt, name, t, dim)


def main():
    map_paths, q_paths, gt, tag = load_data()
    print(f"database={len(map_paths)} queries={len(q_paths)} descriptor={C['descriptor']} kp={C['n_keypoints']}")
    m_raw = extract_all(map_paths, f"map_{tag}")
    q_raw = extract_all(q_paths, f"query_{tag}")
    print("avg keypoints/image (map):", np.mean([len(f[0]) for f in m_raw]).round(1))

    pca = PCA(C["pca_dim"], random_state=C["seed"]).fit(stack(m_raw, 200_000))
    proj = lambda feats: [(pca.transform(d).astype(np.float32) if len(d) else d[:, :C["pca_dim"]], xy, s) for d, xy, s in feats]
    m_f, q_f = proj(m_raw), proj(q_raw)
    sample = stack(m_f, 200_000)

    rows, aps, orders, sims = [], {}, {}, {}

    def register(S, name, t, dim):
        row, ap, order = evaluate(S, gt, name, t, dim)
        rows.append(row); aps[name] = ap; orders[name] = order; sims[name] = S
        print(f"{name:<22} MAP={row['MAP']:.4f}  P@10={row['P@10']:.4f}  F1@10={row['F1@10']:.4f}  dim={dim}")

    def kmeans(k):
        return MiniBatchKMeans(k, random_state=C["seed"], n_init=3, batch_size=4096).fit(sample).cluster_centers_.astype(np.float32)

    t0 = time.time(); cen_bow = kmeans(C["bow_k"]); t_voc_bow = time.time() - t0

    t0 = time.time()
    Hm, Hq = bow_counts(m_f, cen_bow), bow_counts(q_f, cen_bow)
    mu, sd = Hm.mean(0), Hm.std(0) + 1e-9
    Zm, Zq = (Hm - mu) / sd, (Hq - mu) / sd
    S = -np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] - 2 * Zq @ Zm.T + (Zm ** 2).sum(1)[None], 0))
    register(S, "M0 BoW A02 (z+L2dist)", time.time() - t0, C["bow_k"])

    t0 = time.time(); Tm, Tq = tfidf_l2(Hm, Hq); S = Tq @ Tm.T
    register(S, "M1 BoW TF-IDF", time.time() - t0, C["bow_k"])

    t0 = time.time()
    Sm, Sq = bow_counts(m_f, cen_bow, C["soft_n"]), bow_counts(q_f, cen_bow, C["soft_n"])
    Tm, Tq = tfidf_l2(Sm, Sq); S = Tq @ Tm.T
    register(S, f"M2 BoW soft (n={C['soft_n']})", time.time() - t0, C["bow_k"])

    cen_v = kmeans(C["vlad_k"])
    t0 = time.time(); Vm, Vq = vlad_encode(m_f, cen_v), vlad_encode(q_f, cen_v); S = Vq @ Vm.T
    register(S, "M3 VLAD", time.time() - t0, Vm.shape[1])

    t0 = time.time()
    SVm, SVq = vlad_encode(m_f, cen_v, C["grid_levels"]), vlad_encode(q_f, cen_v, C["grid_levels"])
    S = SVq @ SVm.T
    register(S, "M4 spatial VLAD", time.time() - t0, SVm.shape[1])
    del SVm, SVq

    gmm = GaussianMixture(C["fisher_k"], covariance_type="diag", max_iter=50, random_state=C["seed"]).fit(sample[:100_000])
    t0 = time.time(); Fm, Fq = fisher_encode(m_f, gmm), fisher_encode(q_f, gmm); S = Fq @ Fm.T
    register(S, "M5 Fisher vector", time.time() - t0, Fm.shape[1])

    best = max(rows, key=lambda r: r["MAP"])["method"]
    t0 = time.time()
    order_gv = geometric_rerank(orders[best], sims[best], q_raw, m_raw, C["gv_shortlist"])
    row, ap, order = eval_order(order_gv, gt, f"M6 {best.split()[0]}+GV(top{C['gv_shortlist']})", 0, time.time() - t0)
    row["dim"] = rows[[r["method"] for r in rows].index(best)]["dim"]
    rows.append(row); aps[row["method"]] = ap
    print(f"{row['method']:<22} MAP={row['MAP']:.4f}  P@10={row['P@10']:.4f}  F1@10={row['F1@10']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C["out_dir"], "results.csv"), index=False)
    names = list(aps)
    wrows = []
    for ref in (names[0], names[1]):
        for n in names:
            if n == ref:
                continue
            diff = aps[n] - aps[ref]
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            wrows.append(dict(method=n, vs=ref, mean_dAP=diff.mean(), p_value=p, significant=p < 0.05))
    pd.DataFrame(wrows).to_csv(os.path.join(C["out_dir"], "wilcoxon.csv"), index=False)
    np.savez(os.path.join(C["out_dir"], "per_query_ap.npz"), **{k.replace(" ", "_"): v for k, v in aps.items()})

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.barh(df["method"][::-1], df["MAP"][::-1], color="#3b6fb6")
        for i, v in enumerate(df["MAP"][::-1]):
            ax.text(v, i, f" {v:.3f}", va="center")
        ax.set_xlabel("MAP"); ax.set_title(f"Histogram replacement ({C['descriptor']}, {tag} dataset)")
        plt.tight_layout(); plt.savefig(os.path.join(C["out_dir"], "results.png"), dpi=150)
    except Exception as e:
        print("plot skipped:", e)

    pd.set_option("display.width", 200)
    print("\n", df.round(4).to_string(index=False))
    print("\nWilcoxon vs M0 (p<0.05 = significant):")
    print(pd.DataFrame(wrows).query("vs == @names[0]").round(4).to_string(index=False))


if __name__ == "__main__":
    main()