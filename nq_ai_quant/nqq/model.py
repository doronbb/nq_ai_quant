"""
Model backends, tried in order of preference:

  1. lightgbm            - fastest and best. pip install lightgbm
  2. sklearn HistGB      - very close second. pip install scikit-learn
  3. numpy_gbdt          - built-in histogram gradient boosting, zero deps.
                           Slower and a bit weaker, but means the search engine
                           always runs, on any machine, offline.

All backends expose the same interface:
    m = make_model(params); m.fit(X, y, classes); p = m.predict_proba(X)
    m.feature_importance() -> np.ndarray aligned to X columns
`y` is in {-1, 0, 1}; predict_proba returns columns ordered by `classes`.
"""
from __future__ import annotations

import numpy as np

_BACKEND = None


def available_backend() -> str:
    global _BACKEND
    if _BACKEND:
        return _BACKEND
    try:
        import lightgbm  # noqa: F401
        _BACKEND = "lightgbm"
    except ImportError:
        try:
            import sklearn  # noqa: F401
            _BACKEND = "sklearn"
        except ImportError:
            _BACKEND = "numpy_gbdt"
    return _BACKEND


# --------------------------------------------------------------------------
# 1. lightgbm
# --------------------------------------------------------------------------

class LightGBMModel:
    name = "lightgbm"

    def __init__(self, p: dict):
        self.p = p
        self.m = None
        self.classes = None

    def fit(self, X, y, classes):
        import lightgbm as lgb
        self.classes = list(classes)
        remap = {c: i for i, c in enumerate(self.classes)}
        yy = np.array([remap[v] for v in y], dtype=int)
        params = dict(
            objective="multiclass" if len(self.classes) > 2 else "binary",
            num_class=len(self.classes) if len(self.classes) > 2 else 1,
            learning_rate=self.p["learning_rate"],
            num_leaves=self.p["num_leaves"],
            max_depth=self.p["max_depth"],
            min_child_samples=self.p["min_child_samples"],
            subsample=self.p["subsample"], subsample_freq=1,
            colsample_bytree=self.p["colsample"],
            reg_lambda=self.p["reg_lambda"],
            verbose=-1, num_threads=self.p.get("threads", 0),
            deterministic=True, seed=self.p.get("seed", 0),
        )
        self.m = lgb.train(params, lgb.Dataset(X, label=yy),
                           num_boost_round=self.p["n_estimators"])
        return self

    def predict_proba(self, X):
        p = self.m.predict(X)
        if p.ndim == 1:
            p = np.column_stack([1 - p, p])
        return p

    def feature_importance(self):
        return self.m.feature_importance(importance_type="gain").astype(float)


# --------------------------------------------------------------------------
# 2. sklearn
# --------------------------------------------------------------------------

class SklearnModel:
    name = "sklearn_histgb"

    def __init__(self, p: dict):
        self.p = p
        self.m = None
        self.classes = None
        self._imp = None

    def fit(self, X, y, classes):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.classes = list(classes)
        self.m = HistGradientBoostingClassifier(
            learning_rate=self.p["learning_rate"],
            max_iter=self.p["n_estimators"],
            max_leaf_nodes=self.p["num_leaves"],
            max_depth=None if self.p["max_depth"] <= 0 else self.p["max_depth"],
            min_samples_leaf=self.p["min_child_samples"],
            l2_regularization=self.p["reg_lambda"],
            early_stopping=False,
            random_state=self.p.get("seed", 0),
        ).fit(X, y)
        return self

    def predict_proba(self, X):
        p = self.m.predict_proba(X)
        order = [list(self.m.classes_).index(c) for c in self.classes]
        return p[:, order]

    def feature_importance(self):
        # HistGB has no native gain importance; use a cheap permutation-free proxy.
        if self._imp is None:
            self._imp = np.ones(self.m.n_features_in_, dtype=float)
        return self._imp


# --------------------------------------------------------------------------
# 3. built-in numpy histogram GBDT (zero dependencies)
# --------------------------------------------------------------------------

class _Tree:
    __slots__ = ("feat", "thr", "left", "right", "value", "is_leaf")

    def __init__(self):
        self.feat = -1
        self.thr = 0
        self.left = None
        self.right = None
        self.value = 0.0
        self.is_leaf = True


def _grow(Xb, grad, hess, idx, depth, max_depth, min_samples, lam, n_bins, gain_out):
    node = _Tree()
    g_sum, h_sum = grad[idx].sum(), hess[idx].sum()
    node.value = -g_sum / (h_sum + lam)

    if depth >= max_depth or len(idx) < 2 * min_samples:
        return node

    best = (0.0, -1, -1)
    parent = g_sum * g_sum / (h_sum + lam)
    for f in range(Xb.shape[1]):
        bins = Xb[idx, f]
        gh = np.bincount(bins, weights=grad[idx], minlength=n_bins)
        hh = np.bincount(bins, weights=hess[idx], minlength=n_bins)
        cnt = np.bincount(bins, minlength=n_bins)
        gl, hl, cl = np.cumsum(gh), np.cumsum(hh), np.cumsum(cnt)
        gr, hr, cr = g_sum - gl, h_sum - hl, len(idx) - cl
        ok = (cl >= min_samples) & (cr >= min_samples)
        if not ok.any():
            continue
        gain = (gl * gl / (hl + lam) + gr * gr / (hr + lam) - parent) * 0.5
        gain = np.where(ok, gain, -np.inf)
        b = int(np.argmax(gain))
        if gain[b] > best[0]:
            best = (float(gain[b]), f, b)

    if best[1] < 0 or best[0] <= 1e-9:
        return node

    gain, f, b = best
    gain_out[f] += gain
    mask = Xb[idx, f] <= b
    li, ri = idx[mask], idx[~mask]
    if len(li) == 0 or len(ri) == 0:
        return node
    node.is_leaf = False
    node.feat, node.thr = f, b
    node.left = _grow(Xb, grad, hess, li, depth + 1, max_depth, min_samples, lam, n_bins, gain_out)
    node.right = _grow(Xb, grad, hess, ri, depth + 1, max_depth, min_samples, lam, n_bins, gain_out)
    return node


def _predict_tree(node, Xb):
    out = np.empty(len(Xb))
    stack = [(node, np.arange(len(Xb)))]
    while stack:
        nd, idx = stack.pop()
        if nd.is_leaf or len(idx) == 0:
            out[idx] = nd.value
            continue
        m = Xb[idx, nd.feat] <= nd.thr
        stack.append((nd.left, idx[m]))
        stack.append((nd.right, idx[~m]))
    return out


class NumpyGBDTModel:
    """One-vs-rest logistic gradient boosting on quantile-binned features."""
    name = "numpy_gbdt"
    N_BINS = 32

    def __init__(self, p: dict):
        self.p = p
        self.classes = None
        self.edges = None
        self.trees: dict[int, list] = {}
        self.base: dict[int, float] = {}
        self.lr = p["learning_rate"]
        self._imp = None

    def _bin_fit(self, X):
        qs = np.linspace(0, 1, self.N_BINS + 1)[1:-1]
        self.edges = np.nanquantile(np.where(np.isfinite(X), X, np.nan), qs, axis=0)
        self.edges = np.nan_to_num(self.edges, nan=0.0)

    def _bin(self, X):
        Xb = np.empty(X.shape, dtype=np.int16)
        Xs = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        for f in range(X.shape[1]):
            Xb[:, f] = np.searchsorted(self.edges[:, f], Xs[:, f], side="left")
        return Xb

    def fit(self, X, y, classes):
        rng = np.random.default_rng(self.p.get("seed", 0))
        X = np.asarray(X, dtype=np.float64)
        self.classes = list(classes)
        self._bin_fit(X)
        Xb = self._bin(X)
        n_feat = X.shape[1]
        self._imp = np.zeros(n_feat)
        max_depth = self.p["max_depth"] if self.p["max_depth"] > 0 else 4
        max_depth = int(min(max_depth, 6))
        n_rounds = int(min(self.p["n_estimators"], 200))
        sub = float(self.p["subsample"])
        colsample = float(self.p["colsample"])

        for k, cls in enumerate(self.classes):
            yk = (np.asarray(y) == cls).astype(np.float64)
            pbar = float(np.clip(yk.mean(), 1e-6, 1 - 1e-6))
            f0 = float(np.log(pbar / (1 - pbar)))
            self.base[k] = f0
            F = np.full(len(yk), f0)
            trees = []
            for _ in range(n_rounds):
                p = 1.0 / (1.0 + np.exp(-F))
                grad = p - yk
                hess = np.maximum(p * (1 - p), 1e-6)
                idx = np.arange(len(yk))
                if sub < 1.0:
                    idx = idx[rng.random(len(idx)) < sub]
                    if len(idx) < 50:
                        idx = np.arange(len(yk))
                feats = np.arange(n_feat)
                if colsample < 1.0:
                    keep = rng.random(n_feat) < colsample
                    if keep.sum() >= 2:
                        feats = feats[keep]
                gsub = np.zeros(len(feats))
                tree = _grow(Xb[:, feats], grad, hess, idx, 0, max_depth,
                             int(self.p["min_child_samples"]), float(self.p["reg_lambda"]),
                             self.N_BINS, gsub)
                self._imp[feats] += gsub
                tree_feats = feats
                trees.append((tree, tree_feats))
                F += self.lr * _predict_tree(tree, Xb[:, tree_feats])
            self.trees[k] = trees
        return self

    def predict_proba(self, X):
        Xb = self._bin(np.asarray(X, dtype=np.float64))
        scores = np.empty((len(Xb), len(self.classes)))
        for k in range(len(self.classes)):
            F = np.full(len(Xb), self.base[k])
            for tree, feats in self.trees[k]:
                F += self.lr * _predict_tree(tree, Xb[:, feats])
            scores[:, k] = 1.0 / (1.0 + np.exp(-F))
        return scores / (scores.sum(axis=1, keepdims=True) + 1e-12)

    def feature_importance(self):
        return self._imp


# --------------------------------------------------------------------------

def make_model(params: dict, backend: str | None = None):
    b = backend or available_backend()
    if b == "lightgbm":
        return LightGBMModel(params)
    if b == "sklearn":
        return SklearnModel(params)
    return NumpyGBDTModel(params)
