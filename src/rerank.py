import cv2
import numpy as np
import src.config as config

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(x, **kwargs):
        return x


def geometric_rerank(order, q_feats, m_feats, n_shortlist):

    bf = cv2.BFMatcher(cv2.NORM_L2)
    new_order = order.copy()
    for q in tqdm(range(len(order)), desc="geometric verification"):
        qd, qxy, (qw, qh) = q_feats[q]
        short = order[q, :n_shortlist]
        inl = np.zeros(n_shortlist)
        if len(qd) >= config.GV_MIN_INLIERS:
            for j, m in enumerate(short):
                md, mxy, (mw, mh) = m_feats[m]
                if len(md) < 2:
                    continue
                knn = bf.knnMatch(qd, md, k=2)
                good = [a for a, b in (x for x in knn if len(x) == 2) if a.distance < config.GV_RATIO * b.distance]
                if len(good) < config.GV_MIN_INLIERS:
                    continue
                p1 = qxy[[g.queryIdx for g in good]] * [qw, qh]
                p2 = mxy[[g.trainIdx for g in good]] * [mw, mh]
                _, mask = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC,
                                                      ransacReprojThreshold=0.03 * max(mw, mh))
                inl[j] = 0 if mask is None else int(mask.sum())
        sh_order = np.lexsort((np.arange(n_shortlist), -inl))
        new_order[q, :n_shortlist] = short[sh_order]
    return new_order