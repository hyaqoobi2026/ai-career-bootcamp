"""Real coding exercises with runnable tests -> exercises.json

Each exercise is keyed by the task id it attaches to (see build_curriculum.py).
The app runs `starter`-derived student code in Pyodide, then execs `tests`
against the resulting namespace. Every assert that passes is a green line.

Add an exercise: put a new entry in EXERCISES keyed by task id, re-run this,
then run build_app.py.
"""
from __future__ import annotations
import json, pathlib

EXERCISES: dict[str, dict] = {}


def ex(task_id, title, brief, starter, tests, hints, solution):
    EXERCISES[task_id] = {
        "title": title, "brief": brief.strip(), "starter": starter.strip(),
        "tests": tests.strip(), "hints": hints, "solution": solution.strip(),
    }


# ─────────────────────────────────────────────── Week 1, Mon — linear regression
ex(
    "d1-deep", "Linear regression from scratch",
    """
## What you are building

A function that learns the line `y = wx + b` from data, using nothing but numpy
and gradient descent you wrote yourself. No scikit-learn.

## Do this on paper FIRST

You cannot skip this part. The loss is mean squared error:

```
L = (1/n) · Σ (ŷᵢ − yᵢ)²        where  ŷᵢ = w·xᵢ + b
```

Work out `∂L/∂w` and `∂L/∂b` by hand before writing any code. Use the chain
rule. If you cannot derive them, you do not yet understand what gradient descent
is doing, and typing the answer will not fix that.

When you have them, they should look like:

```
∂L/∂w = (2/n) · Σ (ŷᵢ − yᵢ) · xᵢ
∂L/∂b = (2/n) · Σ (ŷᵢ − yᵢ)
```

## The update rule

Each step, move each parameter a small distance *against* its gradient:

```
w ← w − lr · ∂L/∂w
b ← b − lr · ∂L/∂b
```

## Your task

Implement `fit_linear(x, y, lr=0.01, steps=2000)`.

Return `(w, b, losses)` where `losses` is a list containing the loss **before
each update** — so `len(losses) == steps`.

## Success

On data generated from `y = 3x + 2 + noise`, you should recover `w ≈ 3` and
`b ≈ 2`, and the loss should fall steadily.
""",
    """
import numpy as np

def fit_linear(x, y, lr=0.01, steps=2000):
    \"\"\"Fit y = w*x + b by gradient descent.

    x, y : 1-D numpy arrays of the same length
    returns (w, b, losses)
    \"\"\"
    w = 0.0
    b = 0.0
    losses = []

    for _ in range(steps):
        # 1. predict
        # 2. compute loss, append to losses
        # 3. compute gradients dw, db
        # 4. update w and b
        pass

    return w, b, losses
""",
    """
import numpy as np
rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, 200)
y = 3.0 * x + 2.0 + rng.normal(0, 0.1, 200)

w, b, losses = fit_linear(x, y, lr=0.05, steps=3000)

check(len(losses) == 3000, f"losses should have one entry per step (got {len(losses)})")
check(abs(w - 3.0) < 0.1, f"w should be about 3.0, got {w:.4f}")
check(abs(b - 2.0) < 0.1, f"b should be about 2.0, got {b:.4f}")
check(losses[-1] < losses[0], "loss should decrease overall")
check(losses[-1] < 0.05, f"final loss should be small, got {losses[-1]:.4f}")
check(all(losses[i+1] <= losses[i] + 1e-6 for i in range(0, 200)),
      "loss should fall monotonically early on — if it spikes, your lr is too high "
      "or a gradient sign is flipped")

# must generalise, not memorise
xt = np.array([-5.0, 0.0, 5.0])
pred = w * xt + b
expect = 3.0 * xt + 2.0
check(np.allclose(pred, expect, atol=0.3), "predictions should match the true line")
""",
    [
        "Prediction for the whole array at once: `yhat = w * x + b`. No Python loop over samples.",
        "Loss: `np.mean((yhat - y) ** 2)`.",
        "The error term `(yhat - y)` appears in both gradients. Compute it once.",
        "`dw = 2 * np.mean((yhat - y) * x)` and `db = 2 * np.mean(yhat - y)`.",
        "Append the loss BEFORE updating w and b, or you will have steps+1 entries.",
        "If the loss explodes to nan, your learning rate is too large. Try 0.01.",
    ],
    """
import numpy as np

def fit_linear(x, y, lr=0.01, steps=2000):
    w, b = 0.0, 0.0
    losses = []
    n = len(x)
    for _ in range(steps):
        yhat = w * x + b
        err = yhat - y
        losses.append(float(np.mean(err ** 2)))
        dw = 2.0 * np.mean(err * x)
        db = 2.0 * np.mean(err)
        w -= lr * dw
        b -= lr * db
    return w, b, losses
""",
)

# ─────────────────────────────────────────── Week 1, Tue — multivariate + vectorised
ex(
    "d2-deep", "Linear regression, multivariate",
    """
## What changes

One feature becomes many. Instead of a scalar `w` you now have a weight
**vector** `W` with one entry per feature.

```
ŷ = X @ W + b
```

`X` has shape `(n, d)` — n samples, d features. `W` has shape `(d,)`. The `@`
operator is matrix multiplication, and it produces `(n,)` predictions.

## The gradients, in matrix form

Same chain rule, written for vectors:

```
∂L/∂W = (2/n) · Xᵀ @ (ŷ − y)      shape (d,)
∂L/∂b = (2/n) · Σ (ŷ − y)          scalar
```

**Understand why `Xᵀ` appears.** Each weight `Wⱼ` affects the prediction through
column j of X, so its gradient is the error dotted with that column.
Transposing and multiplying does all d of those dot products at once.

## Shapes are the whole game

Print `.shape` obsessively. Most bugs in this exercise are shape bugs, and the
error messages are unhelpful. `X.T @ err` is `(d,n) @ (n,) → (d,)`. If you get
`(n,)` or a scalar, something is transposed wrong.

## Your task

Implement `fit_linear_multi(X, y, lr=0.01, steps=2000)` returning `(W, b, losses)`.

**No Python loop over samples or features.** One matrix operation per step.
""",
    """
import numpy as np

def fit_linear_multi(X, y, lr=0.01, steps=2000):
    \"\"\"Fit y = X @ W + b by gradient descent.

    X : (n, d) numpy array
    y : (n,)  numpy array
    returns (W, b, losses)   W has shape (d,)
    \"\"\"
    n, d = X.shape
    W = np.zeros(d)
    b = 0.0
    losses = []

    for _ in range(steps):
        # predict -> loss -> gradients -> update
        pass

    return W, b, losses
""",
    """
import numpy as np
rng = np.random.default_rng(1)
n, d = 400, 3
X = rng.normal(0, 1, (n, d))
W_true = np.array([2.0, -1.5, 0.5])
b_true = 4.0
y = X @ W_true + b_true + rng.normal(0, 0.05, n)

W, b, losses = fit_linear_multi(X, y, lr=0.1, steps=3000)
W = np.asarray(W)

check(W.shape == (3,), f"W should have shape (3,), got {W.shape}")
check(np.allclose(W, W_true, atol=0.05),
      f"W should be about {W_true}, got {np.round(W, 4)}")
check(abs(b - b_true) < 0.05, f"b should be about {b_true}, got {b:.4f}")
check(len(losses) == 3000, f"expected 3000 losses, got {len(losses)}")
check(losses[-1] < 0.01, f"final loss should be small, got {losses[-1]:.5f}")

# a different number of features must work too
X2 = rng.normal(0, 1, (200, 5))
W2t = np.array([1.0, 0.0, -2.0, 3.0, 0.5])
y2 = X2 @ W2t + 1.0
W2, b2, _ = fit_linear_multi(X2, y2, lr=0.1, steps=4000)
check(np.allclose(np.asarray(W2), W2t, atol=0.1),
      "your code should work for any number of features, not just 3")
""",
    [
        "`yhat = X @ W + b` — broadcasting adds the scalar b to every prediction.",
        "`err = yhat - y` has shape (n,).",
        "`dW = 2 * (X.T @ err) / n` — check the shape is (d,).",
        "`db = 2 * np.mean(err)`.",
        "If W barely moves, your learning rate is too small for this scale. Try 0.1.",
        "Do not use `np.dot` in a loop over columns. One `@` does all of it.",
    ],
    """
import numpy as np

def fit_linear_multi(X, y, lr=0.01, steps=2000):
    n, d = X.shape
    W = np.zeros(d)
    b = 0.0
    losses = []
    for _ in range(steps):
        yhat = X @ W + b
        err = yhat - y
        losses.append(float(np.mean(err ** 2)))
        W -= lr * (2.0 / n) * (X.T @ err)
        b -= lr * 2.0 * np.mean(err)
    return W, b, losses
""",
)

# ────────────────────────────────────────────── Week 1, Wed — logistic regression
ex(
    "d3-deep", "Logistic regression from scratch",
    """
## The change that matters

Linear regression predicts a number. Now you predict a **probability**, so the
output must be squeezed into (0, 1). That is the sigmoid:

```
σ(z) = 1 / (1 + e^(−z))
```

Your model is `ŷ = σ(X @ W + b)`.

## Why the loss changes

Squared error is the wrong loss for probabilities — it is nearly flat when the
model is confidently wrong, so gradients vanish exactly when you need them.

Use **binary cross-entropy**:

```
L = −(1/n) · Σ [ yᵢ·log(ŷᵢ) + (1−yᵢ)·log(1−ŷᵢ) ]
```

This is Shannon's cross-entropy: how surprised your model was by the truth.

## The result worth knowing

Derive `∂L/∂W` and something remarkable happens — the sigmoid derivative and
the log cancel, and you get:

```
∂L/∂W = (1/n) · Xᵀ @ (ŷ − y)
∂L/∂b = (1/n) · Σ (ŷ − y)
```

**Identical in form to linear regression.** Different model, different loss,
same gradient shape. That is not a coincidence and it is worth sitting with.

## Watch out

`log(0)` is `-inf`. Clip your predictions before taking logs.

## Your task

Implement `sigmoid(z)`, `fit_logistic(X, y, lr, steps)` returning
`(W, b, losses)`, and `predict(X, W, b)` returning hard 0/1 labels.
""",
    """
import numpy as np

def sigmoid(z):
    \"\"\"Elementwise 1 / (1 + exp(-z)). Must not overflow for large |z|.\"\"\"
    pass

def fit_logistic(X, y, lr=0.1, steps=3000):
    \"\"\"X:(n,d)  y:(n,) of 0/1  ->  (W, b, losses) with binary cross-entropy.\"\"\"
    n, d = X.shape
    W = np.zeros(d)
    b = 0.0
    losses = []
    for _ in range(steps):
        pass
    return W, b, losses

def predict(X, W, b):
    \"\"\"Return hard 0/1 labels — threshold the probability at 0.5.\"\"\"
    pass
""",
    """
import numpy as np

# --- sigmoid
check(abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-9, "sigmoid(0) should be 0.5")
big = sigmoid(np.array([-800.0, 800.0]))
check(np.all(np.isfinite(big)), "sigmoid must not overflow or produce nan at +/-800")
check(big[0] < 1e-6 and big[1] > 1 - 1e-6, "sigmoid should saturate to 0 and 1")
z = np.array([-2.0, -0.5, 0.5, 2.0])
check(np.all(np.diff(sigmoid(z)) > 0), "sigmoid should be increasing")

# --- fit on separable data
rng = np.random.default_rng(2)
n = 300
X = np.vstack([rng.normal(-2, 1, (n // 2, 2)), rng.normal(2, 1, (n // 2, 2))])
y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

W, b, losses = fit_logistic(X, y, lr=0.1, steps=3000)
check(len(losses) == 3000, f"expected 3000 losses, got {len(losses)}")
check(losses[-1] < losses[0], "cross-entropy should decrease")
check(losses[-1] < 0.1, f"final loss should be well under 0.1, got {losses[-1]:.4f}")

acc = (predict(X, W, b) == y).mean()
check(acc > 0.97, f"accuracy on separable data should be >97%, got {acc:.3f}")

# --- generalises to held-out points
Xh = np.vstack([rng.normal(-2, 1, (60, 2)), rng.normal(2, 1, (60, 2))])
yh = np.concatenate([np.zeros(60), np.ones(60)])
check((predict(Xh, W, b) == yh).mean() > 0.95, "should generalise to unseen points")

# --- predict returns labels, not probabilities
out = np.asarray(predict(X, W, b))
check(set(np.unique(out)).issubset({0, 1, 0.0, 1.0}),
      "predict must return hard 0/1 labels, not probabilities")
""",
    [
        "For a numerically safe sigmoid, clip z first: `z = np.clip(z, -500, 500)`.",
        "Probabilities: `p = sigmoid(X @ W + b)`.",
        "Clip before the log: `p = np.clip(p, 1e-12, 1 - 1e-12)`.",
        "Loss: `-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))`.",
        "Gradient: `dW = (X.T @ (p - y)) / n` — note there is no factor of 2 here.",
        "`predict` is `(sigmoid(X @ W + b) >= 0.5).astype(int)`.",
    ],
    """
import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def fit_logistic(X, y, lr=0.1, steps=3000):
    n, d = X.shape
    W = np.zeros(d)
    b = 0.0
    losses = []
    for _ in range(steps):
        p = sigmoid(X @ W + b)
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        losses.append(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))))
        err = p - y
        W -= lr * (X.T @ err) / n
        b -= lr * np.mean(err)
    return W, b, losses

def predict(X, W, b):
    return (sigmoid(X @ W + b) >= 0.5).astype(int)
""",
)

# ─────────────────────────────────────── Week 1, Thu — linear algebra by hand
ex(
    "d4-deep", "Linear algebra I — build the operations yourself",
    """
## Why you are doing this

You have been using `@` and `.T` without feeling what they do. Today you
implement them in pure Python, then check against numpy. After this, matrix
shapes stop being something you guess at.

## The three operations

**Dot product** of two vectors — multiply elementwise, sum the result. It
measures how much two vectors point the same way.

**Matrix multiply** `A @ B` where A is `(n, m)` and B is `(m, p)`. The result is
`(n, p)`, and entry `[i][j]` is the dot product of **row i of A** with
**column j of B**. The inner dimensions must match — that is the whole rule.

**Transpose** — flip rows and columns. `(n, m)` becomes `(m, n)`.

## Your task

Implement all three in **pure Python** — lists of lists, no numpy inside your
functions. The tests compare your answers against numpy's.

Then answer, in your own notes: why is `A @ B` almost never equal to `B @ A`?
""",
    """
def dot(a, b):
    \"\"\"Dot product of two equal-length lists of numbers. Returns a float.\"\"\"
    pass

def transpose(A):
    \"\"\"A is a list of rows. Return the transpose as a list of rows.\"\"\"
    pass

def matmul(A, B):
    \"\"\"A is (n,m), B is (m,p), both lists of rows. Return (n,p).

    Raise ValueError if the inner dimensions do not match.
    \"\"\"
    pass
""",
    """
import numpy as np

check(abs(dot([1, 2, 3], [4, 5, 6]) - 32) < 1e-9, "dot([1,2,3],[4,5,6]) should be 32")
check(abs(dot([1, 0], [0, 1])) < 1e-9, "perpendicular vectors have dot product 0")

A = [[1, 2, 3], [4, 5, 6]]
check(transpose(A) == [[1, 4], [2, 5], [3, 6]], f"transpose wrong: {transpose(A)}")
check(transpose(transpose(A)) == A, "transposing twice should return the original")

B = [[7, 8], [9, 10], [11, 12]]
got = matmul(A, B)
want = (np.array(A) @ np.array(B)).tolist()
check(got == want, f"matmul wrong: got {got}, numpy says {want}")

rng = np.random.default_rng(3)
M1 = rng.integers(-5, 5, (4, 6)).tolist()
M2 = rng.integers(-5, 5, (6, 3)).tolist()
check(matmul(M1, M2) == (np.array(M1) @ np.array(M2)).tolist(),
      "matmul disagrees with numpy on a random 4x6 @ 6x3")

I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
check(matmul(A, I3) == A, "multiplying by the identity should change nothing")

raised = False
try:
    matmul([[1, 2]], [[1, 2]])       # (1,2) @ (1,2) is illegal
except ValueError:
    raised = True
check(raised, "matmul should raise ValueError when inner dimensions disagree")
""",
    [
        "`dot`: `sum(x * y for x, y in zip(a, b))`.",
        "`transpose`: `[[A[r][c] for r in range(len(A))] for c in range(len(A[0]))]`.",
        "For matmul, get the columns of B by transposing it first — then every entry is a dot product.",
        "Inner dimensions: `len(A[0])` must equal `len(B)`.",
        "matmul result has `len(A)` rows and `len(B[0])` columns.",
    ],
    """
def dot(a, b):
    return float(sum(x * y for x, y in zip(a, b)))

def transpose(A):
    return [[A[r][c] for r in range(len(A))] for c in range(len(A[0]))]

def matmul(A, B):
    if len(A[0]) != len(B):
        raise ValueError(f"inner dimensions disagree: {len(A[0])} vs {len(B)}")
    Bt = transpose(B)
    return [[dot(row, col) for col in Bt] for row in A]
""",
)

# ─────────────────────────────────── Week 1, Fri — closed form vs gradient descent
ex(
    "d5-deep", "Linear algebra II — the closed-form solution",
    """
## There is an exact answer

Gradient descent *approaches* the best line. For linear regression you can also
compute it **exactly**, in one step, with the normal equation:

```
W = (Xᵀ X)⁻¹ Xᵀ y
```

## Why bother, if we have gradient descent?

Two reasons, and the second is the important one.

**It gives you ground truth.** You can now check whether your gradient descent
actually converged to the right answer, instead of hoping.

**It shows you why gradient descent exists.** Inverting `XᵀX` costs roughly
O(d³) for d features. At d = 100 that is nothing. At d = 1,000,000 — which is a
small neural network — it is impossible. Gradient descent is what you use when
the exact answer exists but you cannot afford to compute it.

That is the trade the whole field runs on.

## The intercept

The formula has no `b`. The standard trick: append a column of 1s to X. That
column's weight *is* the intercept, because `1 · b = b`.

## Your task

Implement `normal_equation(X, y)` returning `(W, b)`, where X is `(n, d)`
**without** a bias column — you add it inside.

Use `np.linalg.solve`, not `np.linalg.inv`. Explicitly inverting is slower and
numerically worse; solving the system directly is what you should always reach
for.
""",
    """
import numpy as np

def normal_equation(X, y):
    \"\"\"Exact least-squares fit. X:(n,d) without bias column, y:(n,)

    Return (W, b) where W has shape (d,).
    \"\"\"
    pass
""",
    """
import numpy as np
rng = np.random.default_rng(4)
n, d = 500, 4
X = rng.normal(0, 1, (n, d))
W_true = np.array([1.5, -2.0, 0.75, 3.0])
b_true = -1.25
y = X @ W_true + b_true + rng.normal(0, 0.01, n)

W, b = normal_equation(X, y)
W = np.asarray(W)

check(W.shape == (4,), f"W should have shape (4,), got {W.shape}")
check(np.allclose(W, W_true, atol=0.01), f"W should be about {W_true}, got {np.round(W,4)}")
check(abs(b - b_true) < 0.01, f"b should be about {b_true}, got {b:.4f}")

# must agree with numpy's own least-squares
Xb = np.hstack([X, np.ones((n, 1))])
ref = np.linalg.lstsq(Xb, y, rcond=None)[0]
check(np.allclose(np.append(W, b), ref, atol=1e-6),
      "your solution should match np.linalg.lstsq to high precision")

# exact fit on noiseless data
X2 = rng.normal(0, 1, (50, 2))
y2 = X2 @ np.array([2.0, -1.0]) + 5.0
W2, b2 = normal_equation(X2, y2)
check(np.allclose(np.asarray(W2), [2.0, -1.0], atol=1e-8) and abs(b2 - 5.0) < 1e-8,
      "with no noise the fit should be exact to ~1e-8")
""",
    [
        "Add the bias column: `Xb = np.hstack([X, np.ones((len(X), 1))])`.",
        "Build the system: `A = Xb.T @ Xb` and `rhs = Xb.T @ y`.",
        "Solve it: `theta = np.linalg.solve(A, rhs)` — do not use `np.linalg.inv`.",
        "The last entry of theta is b; everything before it is W.",
        "`W, b = theta[:-1], float(theta[-1])`.",
    ],
    """
import numpy as np

def normal_equation(X, y):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    theta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
    return theta[:-1], float(theta[-1])
""",
)

# ───────────────────────────────────────────────── Week 1, Sat — the retention test
ex(
    "d6-deep", "Week-1 rebuild — from a blank file",
    """
## This is the real test of the week

Everything below, from memory. **Close the other exercises. Do not look at your
earlier code.** If you cannot do this, you did not learn it — you followed it,
which feels identical and is not the same thing.

Rebuild:

1. `fit_linear(x, y, lr, steps)` — single feature, returns `(w, b, losses)`
2. `fit_logistic(X, y, lr, steps)` — returns `(W, b, losses)`, cross-entropy
3. `predict(X, W, b)` — hard 0/1 labels
4. `normal_equation(X, y)` — returns `(W, b)`

You will need `sigmoid` too.

## If you get stuck

Do not open the solution. Derive the gradient on paper again. The gradient is
the thing you are actually learning this week; everything else is typing.

## When it passes

Write one paragraph in your repo: what did you have to look up, and what came
back on its own? That gap is your revision list for next week.
""",
    """
import numpy as np

# From memory. No references.

def fit_linear(x, y, lr=0.05, steps=3000):
    pass

def sigmoid(z):
    pass

def fit_logistic(X, y, lr=0.1, steps=3000):
    pass

def predict(X, W, b):
    pass

def normal_equation(X, y):
    pass
""",
    """
import numpy as np
rng = np.random.default_rng(7)

# 1. linear
x = rng.uniform(-3, 3, 200)
y = -1.5 * x + 4.0 + rng.normal(0, 0.1, 200)
w, b, losses = fit_linear(x, y, lr=0.05, steps=3000)
check(abs(w + 1.5) < 0.1, f"linear w should be about -1.5, got {w:.4f}")
check(abs(b - 4.0) < 0.1, f"linear b should be about 4.0, got {b:.4f}")
check(len(losses) == 3000 and losses[-1] < losses[0], "losses should be recorded and decrease")

# 2. sigmoid
check(abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-9, "sigmoid(0) should be 0.5")
check(np.all(np.isfinite(sigmoid(np.array([-900.0, 900.0])))), "sigmoid must not overflow")

# 3. logistic
n = 300
X = np.vstack([rng.normal(-2, 1, (n//2, 2)), rng.normal(2, 1, (n//2, 2))])
yc = np.concatenate([np.zeros(n//2), np.ones(n//2)])
W, bb, lo = fit_logistic(X, yc, lr=0.1, steps=3000)
check(lo[-1] < 0.1, f"cross-entropy should end under 0.1, got {lo[-1]:.4f}")
check((predict(X, W, bb) == yc).mean() > 0.97, "logistic accuracy should exceed 97%")

# 4. normal equation
Xn = rng.normal(0, 1, (300, 3))
Wt = np.array([2.0, -1.0, 0.5])
yn = Xn @ Wt + 3.0
Wn, bn = normal_equation(Xn, yn)
check(np.allclose(np.asarray(Wn), Wt, atol=1e-6), "normal equation W is wrong")
check(abs(bn - 3.0) < 1e-6, "normal equation b is wrong")

# 5. the two methods must agree
Xg = rng.normal(0, 1, (300, 1))
yg = (Xg[:, 0] * 2.0) + 1.0 + rng.normal(0, 0.01, 300)
wg, bg, _ = fit_linear(Xg[:, 0], yg, lr=0.1, steps=5000)
Wne, bne = normal_equation(Xg, yg)
check(abs(wg - float(np.asarray(Wne)[0])) < 0.02,
      "gradient descent and the normal equation should reach the same answer")
""",
    [
        "Start with the loss. If you can write the loss, the gradient follows.",
        "Squared error for regression; cross-entropy for classification.",
        "Both gradients reduce to X-transpose times the error, divided by n.",
        "If you are truly stuck on one, do the others first and come back.",
    ],
    """
import numpy as np

def fit_linear(x, y, lr=0.05, steps=3000):
    w, b, losses = 0.0, 0.0, []
    for _ in range(steps):
        err = (w * x + b) - y
        losses.append(float(np.mean(err ** 2)))
        w -= lr * 2.0 * np.mean(err * x)
        b -= lr * 2.0 * np.mean(err)
    return w, b, losses

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def fit_logistic(X, y, lr=0.1, steps=3000):
    n, d = X.shape
    W, b, losses = np.zeros(d), 0.0, []
    for _ in range(steps):
        p = sigmoid(X @ W + b)
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        losses.append(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))))
        err = p - y
        W -= lr * (X.T @ err) / n
        b -= lr * np.mean(err)
    return W, b, losses

def predict(X, W, b):
    return (sigmoid(X @ W + b) >= 0.5).astype(int)

def normal_equation(X, y):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    theta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
    return theta[:-1], float(theta[-1])
""",
)


if __name__ == "__main__":
    out = pathlib.Path(__file__).parent / "exercises.json"
    out.write_text(json.dumps(EXERCISES, indent=1))
    print(f"exercises.json — {len(EXERCISES)} exercises")
    for k, v in EXERCISES.items():
        n_checks = v["tests"].count("check(")
        print(f"  {k:<10} {v['title'][:44]:<46} {n_checks} checks, {len(v['hints'])} hints")
