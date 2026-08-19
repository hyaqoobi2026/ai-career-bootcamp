"""Week 4 — classical ML, evaluation, and Gate 1. Imported by build_exercises.py."""


def add(ex):
    # ──────────────────────────────────────────── Mon — k-NN and k-means
    ex(
        "d22-deep", "k-NN and k-means from scratch",
        """
## Two algorithms, one idea

Both rest entirely on **distance**. They are the simplest useful ML algorithms
and they expose the assumption every distance-based method makes: that
closeness in feature space means similarity in reality.

**k-NN** — to classify a point, find its k nearest neighbours in the training
set and take a majority vote. There is no training at all; the "model" is the
data. This is *lazy learning*.

**k-means** — to cluster, pick k centres, assign every point to its nearest
centre, move each centre to the mean of its members, repeat until nothing
changes. Unsupervised: no labels anywhere.

## k-means finds a local optimum, not the best one

This matters and the tests will catch it. Depending on where the initial centres
land, k-means can converge to an obviously wrong answer — two centres splitting
one real cluster while a third centre covers two. It cannot escape, because no
single reassignment improves things from there.

The standard fix is to **run it several times from different starts and keep the
best**, measured by **inertia** — the total squared distance from each point to
its assigned centre. Lower is better. scikit-learn does exactly this, ten times,
by default.

So your `kmeans` takes `n_init` and returns the best run. This is your first
encounter with a non-convex optimisation problem, and restarts are the crudest
and most common defence.

## The curse of dimensionality

In high dimensions, distances stop discriminating — the ratio between the
nearest and farthest point approaches 1, so "nearest neighbour" becomes nearly
meaningless. You will measure this yourself in the last test.

This is why raw high-dimensional data usually gets projected down first, and why
learned embeddings matter: they are a *learned* low-dimensional space where
distance actually means something.

## Your task

`euclidean(a, B)`, `knn_predict(X_train, y_train, X_test, k)`,
`inertia(X, C, labels)`, and `kmeans(X, k, seed, max_iter, n_init)` returning
`(centroids, labels)` from the best of `n_init` restarts.
""",
        """
import numpy as np

def euclidean(a, B):
    \"\"\"Distance from point a:(d,) to every row of B:(n,d). Returns (n,).\"\"\"
    pass

def knn_predict(X_train, y_train, X_test, k=3):
    \"\"\"Majority vote of the k nearest neighbours. Returns (m,) labels.\"\"\"
    pass

def inertia(X, C, labels):
    \"\"\"Total squared distance from each point to its assigned centroid.\"\"\"
    pass

def kmeans(X, k, seed=0, max_iter=100, n_init=10):
    \"\"\"Lloyd's algorithm, restarted n_init times; keep the lowest-inertia run.
    Return (centroids:(k,d), labels:(n,)).\"\"\"
    pass
""",
        """
import numpy as np

d = euclidean(np.array([0.,0.]), np.array([[3.,4.],[0.,0.],[1.,0.]]))
check(np.allclose(d, [5.,0.,1.]), f"euclidean wrong: {d}")

rng = np.random.default_rng(41)
Xa = rng.normal([-3,-3], 0.6, (60,2)); Xb = rng.normal([3,3], 0.6, (60,2))
Xtr = np.vstack([Xa,Xb]); ytr = np.array([0]*60 + [1]*60)

pred = np.asarray(knn_predict(Xtr, ytr, np.array([[-3.,-3.],[3.,3.]]), k=3)).reshape(-1)
check(np.array_equal(pred, [0,1]), f"k-NN should classify the cluster centres: {pred}")
check((np.asarray(knn_predict(Xtr,ytr,Xtr,k=3)).reshape(-1)==ytr).mean()>0.97,
      "k-NN should be near-perfect on its own well-separated training data")

Xte = np.vstack([rng.normal([-3,-3],0.6,(20,2)), rng.normal([3,3],0.6,(20,2))])
yte = np.array([0]*20+[1]*20)
check((np.asarray(knn_predict(Xtr,ytr,Xte,k=5)).reshape(-1)==yte).mean()>0.95,
      "k-NN should generalise to held-out points")

# k=1 must reproduce the training labels exactly
check(np.array_equal(np.asarray(knn_predict(Xtr,ytr,Xtr,k=1)).reshape(-1), ytr),
      "with k=1 every training point is its own nearest neighbour, so accuracy is 100%")

# inertia
check(abs(inertia(np.array([[0.,0.],[2.,0.]]), np.array([[0.,0.],[2.,0.]]),
                  np.array([0,1]))) < 1e-12, "inertia is 0 when points sit on their centroids")
check(abs(inertia(np.array([[0.,0.],[2.,0.]]), np.array([[1.,0.]]),
                  np.array([0,0])) - 2.0) < 1e-9, "inertia should be 1^2 + 1^2 = 2")

# k-means on three blobs
cent_true = np.array([[-5.,0.],[5.,0.],[0.,6.]])
X3 = np.vstack([rng.normal(c,0.5,(50,2)) for c in cent_true])
C, lab = kmeans(X3, 3, seed=0, n_init=10)
C = np.asarray(C); lab = np.asarray(lab).reshape(-1)
check(C.shape == (3,2), f"centroids should be (3,2), got {C.shape}")
check(len(np.unique(lab)) == 3, "all three clusters should be used")

found = sorted([tuple(np.round(c,0)) for c in C])
want  = sorted([tuple(np.round(c,0)) for c in cent_true])
check(all(min(np.linalg.norm(c-t) for c in C) < 0.5 for t in cent_true),
      f"centroids should land near {cent_true.tolist()}, got {np.round(C,2).tolist()}")

# each cluster should be pure
for cl in np.unique(lab):
    members = np.where(lab==cl)[0]
    block = members // 50
    check(len(np.unique(block)) == 1, "each k-means cluster should contain one true blob")

# restarts must actually help — seed 0 alone lands in a bad local optimum here
Cs, ls = kmeans(X3, 3, seed=0, n_init=1)
check(inertia(X3, C, lab) <= inertia(X3, np.asarray(Cs), np.asarray(ls)) + 1e-9,
      "10 restarts must never be worse than 1 — you are not keeping the lowest-inertia run")

# the curse of dimensionality, measured
def contrast(dim):
    P = rng.normal(0,1,(300,dim))
    dd = euclidean(P[0], P[1:])
    return dd.max()/dd.min()
check(contrast(2) > contrast(200),
      "the ratio of farthest to nearest distance should SHRINK as dimension grows — "
      "that is the curse of dimensionality")
""",
        [
            "`euclidean`: `np.sqrt(((B - a)**2).sum(axis=1))`.",
            "For k-NN, `np.argsort(dists)[:k]` gives the nearest indices.",
            "Majority vote: `np.bincount(y_train[idx]).argmax()`.",
            "k-means init: `X[rng.choice(len(X), k, replace=False)]`.",
            "Assign: for each point, the argmin over centroids. Update: `X[labels==j].mean(axis=0)`.",
            "Stop early when the labels stop changing — that is convergence.",
            "inertia: `sum(((X - C[labels])**2).sum(axis=1))`.",
            "For restarts, loop n_init times using seed+i, compute inertia each time, keep the best.",
        ],
        """
import numpy as np

def euclidean(a, B):
    return np.sqrt(((np.asarray(B) - np.asarray(a))**2).sum(axis=1))

def knn_predict(X_train, y_train, X_test, k=3):
    out = []
    for x in np.asarray(X_test):
        idx = np.argsort(euclidean(x, X_train))[:k]
        out.append(np.bincount(np.asarray(y_train)[idx].astype(int)).argmax())
    return np.array(out)

def inertia(X, C, labels):
    X = np.asarray(X); C = np.asarray(C); labels = np.asarray(labels)
    return float(((X - C[labels])**2).sum())

def _kmeans_once(X, k, seed, max_iter):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].astype(float).copy()
    labels = np.zeros(len(X), dtype=int)
    for it in range(max_iter):
        new = np.array([int(np.argmin(euclidean(x, C))) for x in X])
        if it > 0 and np.array_equal(new, labels): break
        labels = new
        for j in range(k):
            if np.any(labels == j): C[j] = X[labels == j].mean(axis=0)
    return C, labels

def kmeans(X, k, seed=0, max_iter=100, n_init=10):
    X = np.asarray(X, dtype=float)
    best = None
    for i in range(n_init):
        C, lab = _kmeans_once(X, k, seed + i, max_iter)
        score = inertia(X, C, lab)
        if best is None or score < best[0]: best = (score, C, lab)
    return best[1], best[2]
""",
    )

    # ──────────────────────────────────────── Tue — decision tree
    ex(
        "d23-deep", "Decision tree from scratch",
        """
## The algorithm

Ask a yes/no question about one feature that best splits the data, then recurse
on each half. Stop when a node is pure or too small.

## Measuring "best"

**Gini impurity** — the probability of misclassifying a randomly drawn element
if you labelled it by the node's class distribution:

```
Gini = 1 − Σ pᵢ²
```

Zero when the node is pure. Maximum (0.5 for two classes) when perfectly mixed.

A split's quality is the **weighted** impurity of its children:

```
Gini_split = (n_left/n)·Gini(left) + (n_right/n)·Gini(right)
```

Choose the split that minimises it. Searching every feature and every candidate
threshold is brute force and, for small data, completely fine.

## Why trees matter beyond trees

A single deep tree overfits badly — it can memorise the training set exactly.
That weakness is precisely why **ensembles** exist: bagging many trees on
resampled data (random forests) or fitting trees sequentially to the previous
errors (boosting). Gradient boosting is still the strongest method on tabular
data, and it is built from exactly this.

## Your task

`gini(y)`, `best_split(X, y)` returning `(feature, threshold, score)`,
`build_tree(X, y, depth, max_depth, min_samples)`, and `tree_predict(tree, X)`.
Represent a leaf as `{"leaf": label}` and an internal node as
`{"f": i, "t": thr, "L": ..., "R": ...}` with `L` for `X[:,f] <= thr`.
""",
        """
import numpy as np

def gini(y):
    \"\"\"Gini impurity of a label array. Returns a float.\"\"\"
    pass

def best_split(X, y):
    \"\"\"Return (feature_index, threshold, weighted_gini) minimising impurity,
    or (None, None, None) if no split improves on the parent.\"\"\"
    pass

def build_tree(X, y, depth=0, max_depth=5, min_samples=2):
    pass

def tree_predict(tree, X):
    \"\"\"Return (n,) predicted labels.\"\"\"
    pass
""",
        """
import numpy as np

check(abs(gini(np.array([1,1,1,1]))) < 1e-12, "a pure node has gini 0")
check(abs(gini(np.array([0,0,1,1])) - 0.5) < 1e-12, "a 50/50 two-class node has gini 0.5")
check(abs(gini(np.array([0,0,0,1])) - 0.375) < 1e-9, "gini([0,0,0,1]) = 1-(0.75^2+0.25^2) = 0.375")
check(gini(np.array([0,1,2])) > gini(np.array([0,0,1])), "more mixing means higher gini")

# a split on feature 0 at ~0 is obviously best here
X = np.array([[-2.,5.],[-1.,3.],[1.,4.],[2.,6.]])
y = np.array([0,0,1,1])
f, t, s = best_split(X, y)
check(f == 0, f"the informative feature is 0, got {f}")
check(-1.0 <= t < 1.0, f"threshold should separate -1 from 1, got {t}")
check(abs(s) < 1e-12, "this split is perfect, so weighted gini should be 0")

tree = build_tree(X, y)
check(np.array_equal(np.asarray(tree_predict(tree, X)).reshape(-1), y),
      "the tree must fit this trivial data exactly")

# harder, but learnable
rng = np.random.default_rng(51)
Xb = rng.normal(0, 1, (300, 4))
yb = ((Xb[:,0] > 0.3) & (Xb[:,1] < 0.5)).astype(int)
tb = build_tree(Xb, yb, max_depth=6)
acc = (np.asarray(tree_predict(tb, Xb)).reshape(-1) == yb).mean()
check(acc > 0.95, f"a depth-6 tree should fit this rule to >95%, got {acc:.3f}")

Xh = rng.normal(0,1,(150,4))
yh = ((Xh[:,0] > 0.3) & (Xh[:,1] < 0.5)).astype(int)
check((np.asarray(tree_predict(tb,Xh)).reshape(-1)==yh).mean() > 0.88,
      "and should generalise reasonably to unseen data")

# depth control must actually work
shallow = build_tree(Xb, yb, max_depth=1)
deep    = build_tree(Xb, yb, max_depth=8)
a_s = (np.asarray(tree_predict(shallow,Xb)).reshape(-1)==yb).mean()
a_d = (np.asarray(tree_predict(deep,Xb)).reshape(-1)==yb).mean()
check(a_d > a_s, f"a deeper tree should fit training data better ({a_d:.3f} vs {a_s:.3f})")

# a pure node must become a leaf
leaf = build_tree(np.array([[1.],[2.]]), np.array([1,1]))
check("leaf" in leaf, "a pure node must be a leaf")
""",
        [
            "`gini`: `_, counts = np.unique(y, return_counts=True); p = counts/len(y); return 1 - (p**2).sum()`.",
            "Candidate thresholds: midpoints between consecutive unique values of a feature.",
            "Skip any split that puts zero samples on one side.",
            "Weighted score: `(nL/n)*gini(yL) + (nR/n)*gini(yR)`.",
            "Stop conditions: pure node, depth >= max_depth, or len(y) < min_samples — return `{'leaf': majority}`.",
            "For prediction, walk the dict per row: `node = node['L'] if x[node['f']] <= node['t'] else node['R']`.",
        ],
        """
import numpy as np

def gini(y):
    y = np.asarray(y)
    if len(y) == 0: return 0.0
    _, c = np.unique(y, return_counts=True)
    p = c / len(y)
    return float(1.0 - (p**2).sum())

def best_split(X, y):
    X = np.asarray(X); y = np.asarray(y)
    n, d = X.shape
    best = (None, None, None); bs = gini(y)
    for f in range(d):
        vals = np.unique(X[:, f])
        for i in range(len(vals) - 1):
            t = (vals[i] + vals[i+1]) / 2.0
            m = X[:, f] <= t
            if m.sum() == 0 or (~m).sum() == 0: continue
            s = m.sum()/n * gini(y[m]) + (~m).sum()/n * gini(y[~m])
            if s < bs - 1e-12:
                bs = s; best = (f, float(t), float(s))
    return best

def build_tree(X, y, depth=0, max_depth=5, min_samples=2):
    X = np.asarray(X); y = np.asarray(y)
    maj = int(np.bincount(y.astype(int)).argmax())
    if len(np.unique(y)) == 1 or depth >= max_depth or len(y) < min_samples:
        return {"leaf": maj}
    f, t, _ = best_split(X, y)
    if f is None: return {"leaf": maj}
    m = X[:, f] <= t
    return {"f": f, "t": t,
            "L": build_tree(X[m], y[m], depth+1, max_depth, min_samples),
            "R": build_tree(X[~m], y[~m], depth+1, max_depth, min_samples)}

def tree_predict(tree, X):
    out = []
    for x in np.asarray(X):
        node = tree
        while "leaf" not in node:
            node = node["L"] if x[node["f"]] <= node["t"] else node["R"]
        out.append(node["leaf"])
    return np.array(out)
""",
    )

    # ─────────────────────────────────────────────── Wed — PCA
    ex(
        "d24-deep", "PCA from scratch",
        """
## What PCA does

Finds the directions along which your data varies most, and lets you keep only
those. It is the classical answer to the curse of dimensionality you measured on
Monday.

## The recipe

1. **Centre** the data — subtract the mean of each feature. This is not optional;
   PCA without centring finds the direction to the origin, not the direction of
   variance.
2. **Covariance matrix** — `C = XᵀX / (n−1)` on the centred data. Entry `(i,j)`
   is how features i and j vary together.
3. **Eigendecomposition** of C. Eigenvectors are the **principal components** —
   the directions. Eigenvalues are how much variance lies along each.
4. **Sort** by eigenvalue descending, keep the top k.
5. **Project**: `X_centred @ components.T`.

## Why eigenvectors

The eigenvectors of the covariance matrix are the axes along which the data,
viewed as a cloud, is longest. The largest eigenvalue is the variance along the
longest axis. That is the whole geometric content — everything else is
bookkeeping.

## Use `eigh`, not `eig`

The covariance matrix is symmetric. `np.linalg.eigh` exploits that: it is
faster, and it guarantees real eigenvalues in ascending order. `eig` can return
complex values from floating-point noise and will cause confusing bugs.

## Your task

`pca_fit(X, k)` returning `(components:(k,d), mean:(d,), explained_ratio:(k,))`,
`pca_transform(X, mean, components)`, and `pca_inverse(Z, mean, components)`.
""",
        """
import numpy as np

def pca_fit(X, k):
    \"\"\"Return (components (k,d), mean (d,), explained_variance_ratio (k,)).
    Components sorted by descending explained variance.\"\"\"
    pass

def pca_transform(X, mean, components):
    \"\"\"Project X onto the components. Returns (n,k).\"\"\"
    pass

def pca_inverse(Z, mean, components):
    \"\"\"Reconstruct back into the original space. Returns (n,d).\"\"\"
    pass
""",
        """
import numpy as np
rng = np.random.default_rng(61)

# data stretched along a known direction
t = rng.normal(0, 3, 400)
X = np.column_stack([t, 0.5*t]) + rng.normal(0, 0.05, (400,2)) + np.array([10., -5.])

comp, mean, ratio = pca_fit(X, 2)
comp = np.asarray(comp); mean = np.asarray(mean); ratio = np.asarray(ratio)

check(comp.shape == (2,2), f"components should be (2,2), got {comp.shape}")
check(np.allclose(mean, X.mean(axis=0)), "mean must be the per-feature mean of X")
check(abs(ratio.sum() - 1.0) < 1e-6, f"explained ratios should sum to 1, got {ratio.sum()}")
check(ratio[0] > ratio[1], "components must be sorted by descending variance")
check(ratio[0] > 0.99, f"nearly all variance is along one direction; got {ratio[0]:.4f}")

# the first component points along [1, 0.5] normalised (sign is arbitrary)
want = np.array([1.0, 0.5]); want = want/np.linalg.norm(want)
check(min(np.linalg.norm(comp[0]-want), np.linalg.norm(comp[0]+want)) < 0.05,
      f"first component should align with [1,0.5] normalised, got {np.round(comp[0],3)}")

check(abs(np.linalg.norm(comp[0]) - 1.0) < 1e-9, "components must be unit vectors")
check(abs(comp[0] @ comp[1]) < 1e-9, "components must be orthogonal")

Z = np.asarray(pca_transform(X, mean, comp))
check(Z.shape == (400,2), f"transform should give (400,2), got {Z.shape}")
check(abs(Z.mean(axis=0)).max() < 1e-9, "projected data should be centred at 0")

Xr = np.asarray(pca_inverse(Z, mean, comp))
check(np.allclose(Xr, X, atol=1e-8), "keeping all components must reconstruct X exactly")

# one component should still reconstruct well here
c1, m1, r1 = pca_fit(X, 1)
Z1 = pca_transform(X, m1, c1)
check(np.asarray(Z1).shape == (400,1), "k=1 should give (400,1)")
err = np.abs(np.asarray(pca_inverse(Z1, m1, c1)) - X).max()
check(err < 0.5, f"1-component reconstruction error should be small here, got {err:.3f}")

# centring genuinely matters
Xoff = X + 1000.0
c2, m2, r2 = pca_fit(Xoff, 1)
check(min(np.linalg.norm(np.asarray(c2)[0]-want), np.linalg.norm(np.asarray(c2)[0]+want)) < 0.05,
      "shifting the data must not change the principal direction — did you centre?")
""",
        [
            "`mean = X.mean(axis=0)` then `Xc = X - mean`.",
            "`C = Xc.T @ Xc / (len(X) - 1)`.",
            "`vals, vecs = np.linalg.eigh(C)` — vals ascending, vecs in COLUMNS.",
            "Reverse for descending: `order = np.argsort(vals)[::-1]`.",
            "Components as rows: `vecs[:, order].T[:k]`.",
            "`explained_ratio = vals[order][:k] / vals.sum()`.",
            "`transform`: `(X - mean) @ components.T`. `inverse`: `Z @ components + mean`.",
        ],
        """
import numpy as np

def pca_fit(X, k):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    Xc = X - mean
    C = Xc.T @ Xc / (len(X) - 1)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals = vals[order]; vecs = vecs[:, order]
    return vecs.T[:k], mean, vals[:k] / vals.sum()

def pca_transform(X, mean, components):
    return (np.asarray(X) - np.asarray(mean)) @ np.asarray(components).T

def pca_inverse(Z, mean, components):
    return np.asarray(Z) @ np.asarray(components) + np.asarray(mean)
""",
    )

    # ─────────────────────────────── Thu — the estimator API + pipeline
    ex(
        "d25-deep", "Build the scikit-learn API yourself",
        """
## Why this exercise instead of importing sklearn

The scheduled task is a scikit-learn sprint — do that in VS Code. Here you build
the *interface* sklearn uses, because understanding the contract is worth more
than memorising the imports, and because every ML library in Python copies it.

## The estimator contract

```
model.fit(X, y)      -> returns self, so calls can chain
model.predict(X)     -> predictions
model.score(X, y)    -> a single number, higher is better
```

Learned attributes end with an underscore: `coef_`, `classes_`. This is not
decoration — it is how sklearn distinguishes what you configured from what was
learned. `n_neighbors` is yours; `classes_` came from the data.

## Why `fit` returns self

So `model.fit(X, y).predict(X)` works, and so pipelines can chain. A small
design decision with large ergonomic consequences.

## The rule that prevents the worst bug in ML

A transformer learns its parameters from **training data only**, then applies
them unchanged to test data. Fitting a scaler on all your data before splitting
leaks test statistics into training and inflates your score. That is
**data leakage**, and it is the most common serious mistake in applied ML.

Your `StandardScaler` must store the training mean and standard deviation and
reuse them.

## Your task

`StandardScaler` with `fit`/`transform`/`fit_transform`, and
`KNNClassifier(n_neighbors)` with `fit`/`predict`/`score`, plus `train_test_split`.
""",
        """
import numpy as np

class StandardScaler:
    def fit(self, X):
        \"\"\"Store mean_ and scale_ from X. Return self.\"\"\"
        pass
    def transform(self, X):
        \"\"\"Apply the STORED statistics. Do not recompute.\"\"\"
        pass
    def fit_transform(self, X):
        pass

class KNNClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
    def fit(self, X, y):
        \"\"\"Store the data and set classes_. Return self.\"\"\"
        pass
    def predict(self, X):
        pass
    def score(self, X, y):
        \"\"\"Mean accuracy.\"\"\"
        pass

def train_test_split(X, y, test_size=0.25, seed=0):
    \"\"\"Shuffle then split. Return X_train, X_test, y_train, y_test.\"\"\"
    pass
""",
        """
import numpy as np
rng = np.random.default_rng(71)

X = rng.normal(5, 3, (200, 4))
sc = StandardScaler()
out = sc.fit(X)
check(out is sc, "fit must return self so calls can chain")
check(hasattr(sc, "mean_") and hasattr(sc, "scale_"),
      "learned attributes must be named mean_ and scale_ with trailing underscores")
Xs = np.asarray(sc.transform(X))
check(abs(Xs.mean(axis=0)).max() < 1e-9, "transformed data should have mean 0")
check(abs(Xs.std(axis=0) - 1).max() < 1e-6, "transformed data should have std 1")

# THE LEAKAGE TEST: transform must reuse training statistics
Xtest = rng.normal(50, 3, (50, 4))
Xt = np.asarray(sc.transform(Xtest))
check(abs(Xt.mean(axis=0)).max() > 5,
      "transform must apply the TRAINING mean, not recompute on new data — "
      "if this passes near 0 you have data leakage")

check(np.allclose(np.asarray(StandardScaler().fit_transform(X)), Xs),
      "fit_transform should equal fit then transform")

# knn
Xa = rng.normal([-3,-3], 0.7, (100,2)); Xb = rng.normal([3,3], 0.7, (100,2))
Xk = np.vstack([Xa,Xb]); yk = np.array([0]*100 + [1]*100)
Xtr, Xte, ytr, yte = train_test_split(Xk, yk, test_size=0.25, seed=0)
check(len(Xtr)==150 and len(Xte)==50, f"split sizes wrong: {len(Xtr)}/{len(Xte)}")
check(len(ytr)==150 and len(yte)==50, "y must be split the same way as X")
check(abs(np.mean(yte) - 0.5) < 0.25, "the split must be shuffled, not the first 150 rows")

m = KNNClassifier(n_neighbors=5)
check(m.fit(Xtr, ytr) is m, "fit must return self")
check(set(np.asarray(m.classes_).tolist()) == {0,1}, "classes_ must hold the sorted unique labels")
check(m.score(Xte, yte) > 0.95, f"kNN should score >0.95 here, got {m.score(Xte,yte):.3f}")
check(np.asarray(m.predict(Xte)).shape == (50,), "predict should return one label per row")

# chaining must work
check(KNNClassifier(3).fit(Xtr, ytr).score(Xte, yte) > 0.9,
      "fit().score() chaining should work")

# scaling must not break accuracy on already-comparable features
s2 = StandardScaler().fit(Xtr)
check(KNNClassifier(5).fit(s2.transform(Xtr), ytr).score(s2.transform(Xte), yte) > 0.9,
      "the scaled pipeline should still classify well")
""",
        [
            "`self.mean_ = X.mean(axis=0)`, `self.scale_ = X.std(axis=0)`; guard against zero std with `np.where(s==0, 1, s)`.",
            "`transform` must use `self.mean_` and `self.scale_` — never recompute from its argument.",
            "Every `fit` ends with `return self`.",
            "`self.classes_ = np.unique(y)`.",
            "`score` is `(self.predict(X) == y).mean()`.",
            "For the split: `idx = rng.permutation(len(X))`, then slice at `int(len(X)*(1-test_size))`.",
        ],
        """
import numpy as np

class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        s = X.std(axis=0)
        self.scale_ = np.where(s == 0, 1.0, s)
        return self
    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class KNNClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        return self
    def predict(self, X):
        out = []
        for x in np.asarray(X, dtype=float):
            dist = np.sqrt(((self.X_ - x)**2).sum(axis=1))
            idx = np.argsort(dist)[:self.n_neighbors]
            out.append(np.bincount(self.y_[idx].astype(int)).argmax())
        return np.array(out)
    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())

def train_test_split(X, y, test_size=0.25, seed=0):
    X = np.asarray(X); y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]
""",
    )

    # ─────────────────────────────────────── Fri — evaluation metrics
    ex(
        "d26-deep", "Evaluation metrics — implement every one yourself",
        """
## Why accuracy is usually a lie

A dataset that is 99% negative gives 99% accuracy to a model that always
predicts negative and has learned nothing. Any time classes are imbalanced —
fraud, disease, human rights violations — accuracy is actively misleading.

## The four cells

Everything comes from the confusion matrix:

|                | predicted 0 | predicted 1 |
|----------------|-------------|-------------|
| **actual 0**   | TN          | FP          |
| **actual 1**   | FN          | TP          |

**Precision** = TP/(TP+FP) — of the things I flagged, how many were real?
**Recall** = TP/(TP+FN) — of the real things, how many did I catch?
**F1** = harmonic mean of the two.

## Which one you choose is a values question, not a technical one

Optimise recall when a miss is worse than a false alarm — screening for disease,
or flagging a possible human rights violation where a human reviews everything
flagged anyway. Optimise precision when a false alarm is expensive or harmful.

**Being able to say which you chose and why is one of the strongest signals you
can give in an interview.** Most candidates recite the formulas and never say
what they optimised for.

## ROC-AUC

Sweep the threshold from 1 to 0 and plot true-positive rate against
false-positive rate. The area underneath has a beautiful interpretation:
**the probability that a randomly chosen positive is scored higher than a
randomly chosen negative.** 0.5 is random guessing; 1.0 is perfect ranking.

It is threshold-independent, which makes it a measure of *ranking* quality
rather than of any particular decision rule.

## Your task

`confusion_matrix`, `precision`, `recall`, `f1`, `accuracy`, and `roc_auc`.
Handle division by zero by returning 0.0.
""",
        """
import numpy as np

def confusion_matrix(y_true, y_pred):
    \"\"\"Return (tn, fp, fn, tp) as ints.\"\"\"
    pass

def accuracy(y_true, y_pred): pass
def precision(y_true, y_pred): pass
def recall(y_true, y_pred): pass
def f1(y_true, y_pred): pass

def roc_auc(y_true, scores):
    \"\"\"Area under the ROC curve. y_true is 0/1, scores are continuous.\"\"\"
    pass
""",
        """
import numpy as np

yt = np.array([0,0,0,0,1,1,1,1])
yp = np.array([0,0,1,1,0,1,1,1])
tn, fp, fn, tp = confusion_matrix(yt, yp)
check((tn,fp,fn,tp) == (2,2,1,3), f"confusion matrix wrong: got {(tn,fp,fn,tp)}, want (2,2,1,3)")

check(abs(accuracy(yt,yp) - 5/8) < 1e-9, "accuracy = (tn+tp)/n = 0.625")
check(abs(precision(yt,yp) - 3/5) < 1e-9, "precision = tp/(tp+fp) = 0.6")
check(abs(recall(yt,yp) - 3/4) < 1e-9, "recall = tp/(tp+fn) = 0.75")
check(abs(f1(yt,yp) - 2*0.6*0.75/(0.6+0.75)) < 1e-9, "f1 is the harmonic mean")

# the imbalance trap, made concrete
yi = np.array([0]*99 + [1])
lazy = np.zeros(100, dtype=int)
check(abs(accuracy(yi, lazy) - 0.99) < 1e-9, "the always-negative model gets 99% accuracy")
check(abs(recall(yi, lazy)) < 1e-12, "...and 0% recall — which is what actually matters here")
check(abs(precision(yi, lazy)) < 1e-12, "precision must return 0.0 rather than divide by zero")
check(abs(f1(yi, lazy)) < 1e-12, "f1 must be 0.0, not nan")

# perfect and inverted
check(abs(f1(yt, yt) - 1.0) < 1e-9, "a perfect prediction has f1 = 1")
check(abs(recall(np.array([1,1]), np.array([0,0]))) < 1e-12, "no true positives means recall 0")

# roc-auc
check(abs(roc_auc(np.array([0,0,1,1]), np.array([0.1,0.2,0.8,0.9])) - 1.0) < 1e-9,
      "perfectly ranked scores give AUC 1.0")
check(abs(roc_auc(np.array([0,0,1,1]), np.array([0.9,0.8,0.2,0.1]))) < 1e-9,
      "perfectly inverted ranking gives AUC 0.0")
check(abs(roc_auc(np.array([0,1,0,1]), np.array([0.5,0.5,0.5,0.5])) - 0.5) < 1e-9,
      "all-equal scores give AUC 0.5 — ties count as half")

rng = np.random.default_rng(81)
y = np.array([0]*400 + [1]*400)
s = np.concatenate([rng.normal(0,1,400), rng.normal(2,1,400)])
a = roc_auc(y, s)
check(0.85 < a < 0.96, f"two normals 2 sd apart should give AUC around 0.92, got {a:.3f}")
check(abs(roc_auc(y, s*3 + 100) - a) < 1e-9,
      "AUC depends only on the RANKING — a monotonic rescale must not change it")
""",
        [
            "`tp = int(((y_true==1)&(y_pred==1)).sum())`, and similarly for the other three.",
            "Guard every division: `return tp/(tp+fp) if (tp+fp) > 0 else 0.0`.",
            "f1 is `2*p*r/(p+r)` — and must return 0.0 when p+r is 0.",
            "For AUC, the rank-based formula is easiest and handles ties correctly.",
            "Use `scipy`-free ranking: `order = np.argsort(scores)`, then assign average ranks to tied groups.",
            "AUC = (sum of ranks of positives − n_pos(n_pos+1)/2) / (n_pos · n_neg).",
        ],
        """
import numpy as np

def confusion_matrix(y_true, y_pred):
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    tn = int(((yt==0)&(yp==0)).sum()); fp = int(((yt==0)&(yp==1)).sum())
    fn = int(((yt==1)&(yp==0)).sum()); tp = int(((yt==1)&(yp==1)).sum())
    return tn, fp, fn, tp

def accuracy(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    n = tn+fp+fn+tp
    return float((tn+tp)/n) if n else 0.0

def precision(y_true, y_pred):
    _, fp, _, tp = confusion_matrix(y_true, y_pred)
    return float(tp/(tp+fp)) if (tp+fp) else 0.0

def recall(y_true, y_pred):
    _, _, fn, tp = confusion_matrix(y_true, y_pred)
    return float(tp/(tp+fn)) if (tp+fn) else 0.0

def f1(y_true, y_pred):
    p = precision(y_true, y_pred); r = recall(y_true, y_pred)
    return float(2*p*r/(p+r)) if (p+r) else 0.0

def roc_auc(y_true, scores):
    y = np.asarray(y_true); s = np.asarray(scores, dtype=float)
    n_pos = int((y==1).sum()); n_neg = int((y==0).sum())
    if n_pos == 0 or n_neg == 0: return 0.0
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j+1 < len(s) and s[order[j+1]] == s[order[i]]: j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j+1): ranks[order[k]] = avg
        i = j + 1
    return float((ranks[y==1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg))
""",
    )

    # ────────────────────────────────────────────────── Sat — GATE 1
    ex(
        "d27-deep", "GATE 1 — the month-one assessment",
        """
## Rules

**Three hours. No references. No notes. No previous code. No solution tab.**

If you open the solution, you have not passed the gate — you have read an
answer. The point of an assessment is information about where you actually are,
and a result you engineered is worthless information.

A bad result today is genuinely fine. It tells you exactly what to redo in week
five, which is worth far more than a green tick you did not earn.

## What to build, from nothing

1. `fit_linear(x, y, lr, steps)` — single feature, `(w, b, losses)`
2. `sigmoid(z)` — overflow-safe
3. `fit_logistic(X, y, lr, steps)` — `(W, b, losses)`, cross-entropy
4. `init_params(d, h, seed)` / `forward(X, p)` / `backward(X, y, p, cache)` —
   a 2-layer net with tanh hidden and sigmoid output
5. `train_nn(X, y, h, steps, lr, seed)` — must solve XOR
6. `precision(y_true, y_pred)` and `recall(y_true, y_pred)`

## How you are graded

Backprop is verified by **numerical gradient checking** — every partial
derivative, to 1e-5. Approximately right will fail.

XOR must be solved exactly.

## After

Whatever the result, write in your repo: which parts came back immediately,
which you had to reconstruct, and which you could not do. That list is your
week-five plan.
""",
        """
import numpy as np

# Three hours. From memory. Nothing open.

def fit_linear(x, y, lr=0.05, steps=3000): pass
def sigmoid(z): pass
def fit_logistic(X, y, lr=0.1, steps=3000): pass
def init_params(d, h, seed=0): pass
def forward(X, p): pass
def backward(X, y, p, cache): pass
def train_nn(X, y, h=4, steps=6000, lr=0.5, seed=0): pass
def predict_nn(X, p): pass
def precision(y_true, y_pred): pass
def recall(y_true, y_pred): pass
""",
        """
import numpy as np
rng = np.random.default_rng(99)

# 1 — linear regression
x = rng.uniform(-3,3,250); y = 2.5*x - 1.0 + rng.normal(0,0.1,250)
w,b,L = fit_linear(x,y,lr=0.05,steps=3000)
check(abs(w-2.5)<0.1 and abs(b+1.0)<0.1, f"linear: expected w=2.5 b=-1.0, got {w:.3f} {b:.3f}")
check(len(L)==3000 and L[-1]<L[0], "linear: losses must be recorded and decrease")

# 2 — sigmoid
check(abs(sigmoid(np.array([0.0]))[0]-0.5)<1e-9, "sigmoid(0)=0.5")
check(np.all(np.isfinite(sigmoid(np.array([-900.,900.])))), "sigmoid must not overflow")

# 3 — logistic regression
Xl = np.vstack([rng.normal(-2,1,(120,2)), rng.normal(2,1,(120,2))])
yl = np.concatenate([np.zeros(120), np.ones(120)])
W,bb,Ll = fit_logistic(Xl,yl,lr=0.1,steps=3000)
check(Ll[-1] < 0.15, f"logistic: cross-entropy should end under 0.15, got {Ll[-1]:.4f}")

# 4 — GRADIENT CHECK on the network
p = init_params(2,5,seed=0)
Xg = rng.normal(0,1,(20,2)); yg = (rng.random((20,1))>0.5).astype(float)
A2,cache = forward(Xg,p)
check(np.asarray(A2).shape==(20,1) and np.all((A2>0)&(A2<1)),
      "forward must return (n,1) probabilities")
def _loss(pp):
    a = np.clip(forward(Xg,pp)[0],1e-12,1-1e-12)
    return float(-np.mean(yg*np.log(a)+(1-yg)*np.log(1-a)))
g = backward(Xg,yg,p,cache); bad=[]
for key,gk in [("W1","dW1"),("b1","db1"),("W2","dW2"),("b2","db2")]:
    for i in range(np.asarray(p[key]).size):
        pp={k:v.copy() for k,v in p.items()}
        pp[key].flat[i]+=1e-6; a=_loss(pp)
        pp[key].flat[i]-=2e-6; c=_loss(pp)
        num=(a-c)/2e-6; ana=float(np.asarray(g[gk]).flat[i])
        if abs(num-ana)>1e-5*max(1.0,abs(num)): bad.append(f"{gk}[{i}]")
check(not bad, f"GRADIENT CHECK FAILED on {bad[:5]} — backprop is not exact")

# 5 — XOR
Xx=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]); yx=np.array([[0.],[1.],[1.],[0.]])
px,lx = train_nn(Xx,yx,h=4,steps=6000,lr=0.5,seed=0)
check(lx[-1]<0.05, f"XOR loss should end under 0.05, got {lx[-1]:.4f}")
check(np.array_equal(np.asarray(predict_nn(Xx,px)).reshape(-1),[0,1,1,0]),
      "XOR must be solved exactly")

# 6 — metrics
yt=np.array([0,0,0,0,1,1,1,1]); yp=np.array([0,0,1,1,0,1,1,1])
check(abs(precision(yt,yp)-0.6)<1e-9, "precision = 3/5")
check(abs(recall(yt,yp)-0.75)<1e-9, "recall = 3/4")
check(abs(precision(np.array([0,0]),np.array([0,0])))<1e-12,
      "precision must return 0.0, not nan, when nothing is predicted positive")
""",
        [
            "There are no hints for a gate. That is the point.",
            "If you are stuck, write down what you DO remember and work forward from the loss.",
            "Every gradient in this month reduced to input.T @ error. Start there.",
        ],
        """
# Do not read this before attempting the gate.
import numpy as np

def fit_linear(x, y, lr=0.05, steps=3000):
    w,b,L = 0.0,0.0,[]
    for _ in range(steps):
        e = (w*x+b)-y; L.append(float(np.mean(e**2)))
        w -= lr*2*np.mean(e*x); b -= lr*2*np.mean(e)
    return w,b,L

def sigmoid(z): return 1.0/(1.0+np.exp(-np.clip(z,-500,500)))

def fit_logistic(X, y, lr=0.1, steps=3000):
    n,d = X.shape; W,b,L = np.zeros(d),0.0,[]
    for _ in range(steps):
        p = sigmoid(X@W+b); pc = np.clip(p,1e-12,1-1e-12)
        L.append(float(-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc))))
        e = p-y; W -= lr*(X.T@e)/n; b -= lr*np.mean(e)
    return W,b,L

def init_params(d,h,seed=0):
    rng=np.random.default_rng(seed)
    return {"W1":rng.normal(0,0.5,(d,h)),"b1":np.zeros(h),
            "W2":rng.normal(0,0.5,(h,1)),"b2":np.zeros(1)}

def forward(X,p):
    Z1=X@p["W1"]+p["b1"]; A1=np.tanh(Z1)
    Z2=A1@p["W2"]+p["b2"]; A2=sigmoid(Z2)
    return A2,{"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}

def backward(X,y,p,cache):
    n=X.shape[0]; A1,A2=cache["A1"],cache["A2"]
    dZ2=A2-y
    dW2=A1.T@dZ2/n; db2=np.mean(dZ2,axis=0)
    dZ1=(dZ2@p["W2"].T)*(1-A1**2)
    dW1=X.T@dZ1/n; db1=np.mean(dZ1,axis=0)
    return {"dW1":dW1,"db1":db1,"dW2":dW2,"db2":db2}

def train_nn(X,y,h=4,steps=6000,lr=0.5,seed=0):
    p=init_params(X.shape[1],h,seed); L=[]
    for _ in range(steps):
        A2,c=forward(X,p); a=np.clip(A2,1e-12,1-1e-12)
        L.append(float(-np.mean(y*np.log(a)+(1-y)*np.log(1-a))))
        g=backward(X,y,p,c)
        for k in ("W1","b1","W2","b2"): p[k]=p[k]-lr*g["d"+k]
    return p,L

def predict_nn(X,p): return (forward(X,p)[0]>=0.5).astype(int)

def precision(y_true,y_pred):
    yt,yp=np.asarray(y_true),np.asarray(y_pred)
    tp=int(((yt==1)&(yp==1)).sum()); fp=int(((yt==0)&(yp==1)).sum())
    return float(tp/(tp+fp)) if (tp+fp) else 0.0

def recall(y_true,y_pred):
    yt,yp=np.asarray(y_true),np.asarray(y_pred)
    tp=int(((yt==1)&(yp==1)).sum()); fn=int(((yt==1)&(yp==0)).sum())
    return float(tp/(tp+fn)) if (tp+fn) else 0.0
""",
    )
