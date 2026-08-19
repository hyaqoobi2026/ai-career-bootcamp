"""Week 2 — neural network from scratch. Imported by build_exercises.py."""


def add(ex):
    # ───────────────────────────────────────────────── Mon — forward pass
    ex(
        "d8-deep", "Neuron + forward pass",
        """
## The whole network, forward

One neuron: multiply inputs by weights, add a bias, squash. A layer is many
neurons at once, which is a matrix multiply. A network is layers stacked.

For a 2-layer net with a hidden layer of size `h`:

```
Z1 = X @ W1 + b1        (n, h)
A1 = tanh(Z1)           (n, h)
Z2 = A1 @ W2 + b2       (n, 1)
A2 = sigmoid(Z2)        (n, 1)   -> probability
```

## Shapes are the only hard part

`X` is `(n, d)`. `W1` is `(d, h)`. `W2` is `(h, 1)`. Print every shape until the
chain is obvious. Nearly every bug this week is a shape bug.

## Why initialisation is not zeros

If every weight starts at zero, every hidden neuron computes the same thing and
receives the same gradient forever — they never differentiate. This is the
**symmetry breaking** problem. Initialise with small random values.

## Your task

`init_params(d, h, seed)` returning a dict with `W1, b1, W2, b2`, and
`forward(X, p)` returning `(A2, cache)` where cache holds what backprop will
need next lesson: `Z1, A1, Z2, A2`.
""",
        """
import numpy as np

def init_params(d, h, seed=0):
    \"\"\"Small random weights, zero biases. Return dict with W1,b1,W2,b2.
    W1:(d,h)  b1:(h,)  W2:(h,1)  b2:(1,)\"\"\"
    rng = np.random.default_rng(seed)
    pass

def forward(X, p):
    \"\"\"X:(n,d) -> (A2, cache).  A2:(n,1) probabilities.\"\"\"
    pass
""",
        """
import numpy as np
p = init_params(3, 4, seed=0)
for k, shape in [("W1",(3,4)), ("b1",(4,)), ("W2",(4,1)), ("b2",(1,))]:
    check(k in p, f"params must contain {k}")
    check(np.asarray(p[k]).shape == shape, f"{k} should be {shape}, got {np.asarray(p[k]).shape}")

check(not np.allclose(p["W1"], 0), "W1 must not be all zeros — that breaks symmetry breaking")
check(np.allclose(p["b1"], 0), "biases should start at zero")
check(np.abs(p["W1"]).max() < 1.0, "initial weights should be small (< 1.0)")

check(np.allclose(init_params(3,4,seed=1)["W1"], init_params(3,4,seed=1)["W1"]),
      "same seed must give the same weights — reproducibility matters")
check(not np.allclose(init_params(3,4,seed=1)["W1"], init_params(3,4,seed=2)["W1"]),
      "different seeds should give different weights")

X = np.random.default_rng(5).normal(0, 1, (10, 3))
A2, cache = forward(X, p)
check(np.asarray(A2).shape == (10,1), f"A2 should be (10,1), got {np.asarray(A2).shape}")
check(np.all((A2 > 0) & (A2 < 1)), "A2 must be probabilities strictly in (0,1)")
for k, shape in [("Z1",(10,4)), ("A1",(10,4)), ("Z2",(10,1)), ("A2",(10,1))]:
    check(k in cache, f"cache must contain {k}")
    check(np.asarray(cache[k]).shape == shape, f"cache['{k}'] should be {shape}")
check(np.all(np.abs(cache["A1"]) <= 1.0), "tanh output must lie in [-1, 1]")
check(np.allclose(cache["A1"], np.tanh(cache["Z1"])), "A1 should be tanh(Z1)")
""",
        [
            "`rng.normal(0, 0.1, (d, h))` gives small random weights.",
            "`b1 = np.zeros(h)` and `b2 = np.zeros(1)`.",
            "`Z1 = X @ p['W1'] + p['b1']` — broadcasting adds b1 to every row.",
            "`A1 = np.tanh(Z1)`, then `Z2 = A1 @ p['W2'] + p['b2']`.",
            "sigmoid: `1 / (1 + np.exp(-np.clip(Z2, -500, 500)))`.",
            "Return the cache as a dict so the next exercise can use it.",
        ],
        """
import numpy as np

def init_params(d, h, seed=0):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, 0.1, (d, h)), "b1": np.zeros(h),
            "W2": rng.normal(0, 0.1, (h, 1)), "b2": np.zeros(1)}

def forward(X, p):
    Z1 = X @ p["W1"] + p["b1"]
    A1 = np.tanh(Z1)
    Z2 = A1 @ p["W2"] + p["b2"]
    A2 = 1.0 / (1.0 + np.exp(-np.clip(Z2, -500, 500)))
    return A2, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
""",
    )

    # ─────────────────────────────────────────── Tue — backprop by hand
    ex(
        "d9-deep", "Manual backprop, one hidden layer",
        """
## The hardest day of the month

Do the derivation on paper. All of it. This is the day the whole field either
becomes mechanical to you or stays magic.

## The chain rule, layer by layer

Start at the loss and walk backwards. With binary cross-entropy on a sigmoid
output, the first step collapses beautifully:

```
dZ2 = A2 − y                       (n,1)     <- sigmoid + BCE cancel
dW2 = A1.T @ dZ2 / n               (h,1)
db2 = mean(dZ2)                    (1,)

dA1 = dZ2 @ W2.T                   (n,h)     <- push error back through W2
dZ1 = dA1 * (1 − A1²)              (n,h)     <- tanh'(z) = 1 − tanh(z)²
dW1 = X.T @ dZ1 / n                (d,h)
db1 = mean(dZ1, axis=0)            (h,)
```

## Read the pattern

Every layer does the same three things: **turn the incoming gradient into dZ**
by multiplying by the local derivative, **compute dW** as `input.T @ dZ`, and
**pass dA back** as `dZ @ W.T`.

That pattern is all backpropagation is. Once you see it, deep networks are just
this repeated.

## Your task

`backward(X, y, p, cache)` returning a dict of `dW1, db1, dW2, db2`.
`y` has shape `(n,1)`.
""",
        """
import numpy as np

def backward(X, y, p, cache):
    \"\"\"Return {'dW1','db1','dW2','db2'} with shapes matching the parameters.\"\"\"
    n = X.shape[0]
    A1, A2 = cache["A1"], cache["A2"]
    pass
""",
        """
import numpy as np

def _init(d,h,seed=0):
    rng=np.random.default_rng(seed)
    return {"W1":rng.normal(0,.5,(d,h)),"b1":np.zeros(h),
            "W2":rng.normal(0,.5,(h,1)),"b2":np.zeros(1)}
def _fwd(X,p):
    Z1=X@p["W1"]+p["b1"]; A1=np.tanh(Z1)
    Z2=A1@p["W2"]+p["b2"]; A2=1/(1+np.exp(-np.clip(Z2,-500,500)))
    return A2,{"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
def _loss(A2,y):
    p=np.clip(A2,1e-12,1-1e-12)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

rng=np.random.default_rng(11)
X=rng.normal(0,1,(25,3)); y=(rng.random((25,1))>0.5).astype(float)
p=_init(3,4,seed=0); A2,cache=_fwd(X,p)
g=backward(X,y,p,cache)

for k,shape in [("dW1",(3,4)),("db1",(4,)),("dW2",(4,1)),("db2",(1,))]:
    check(k in g, f"gradients must contain {k}")
    check(np.asarray(g[k]).shape==shape,
          f"{k} must match its parameter shape {shape}, got {np.asarray(g[k]).shape}")

# THE REAL TEST: numerical gradient checking.
def numgrad(key, idx, eps=1e-6):
    pp={k:v.copy() for k,v in p.items()}
    pp[key].flat[idx]+=eps; lp=_loss(_fwd(X,pp)[0],y)
    pp[key].flat[idx]-=2*eps; lm=_loss(_fwd(X,pp)[0],y)
    return (lp-lm)/(2*eps)

bad=[]
for key,gk in [("W1","dW1"),("b1","db1"),("W2","dW2"),("b2","db2")]:
    for idx in range(np.asarray(p[key]).size):
        num=numgrad(key,idx); ana=float(np.asarray(g[gk]).flat[idx])
        if abs(num-ana) > 1e-5*max(1.0,abs(num)):
            bad.append(f"{gk}[{idx}]: analytic {ana:.8f} vs numerical {num:.8f}")
check(not bad, "gradients disagree with numerical differentiation — "
      + (bad[0] if bad else "") + (f" (+{len(bad)-1} more)" if len(bad)>1 else ""))

check(len(bad)==0, "every single partial derivative must match to 1e-5")
""",
        [
            "`dZ2 = (A2 - y)` — this is the sigmoid and cross-entropy derivatives already combined.",
            "`dW2 = A1.T @ dZ2 / n`. Check the shape is (h,1).",
            "`db2 = np.mean(dZ2, axis=0)` gives shape (1,).",
            "`dA1 = dZ2 @ p['W2'].T` has shape (n,h).",
            "tanh derivative: `dZ1 = dA1 * (1 - A1**2)`. Use A1, not Z1 — cheaper and equal.",
            "`dW1 = X.T @ dZ1 / n`, `db1 = np.mean(dZ1, axis=0)`.",
        ],
        """
import numpy as np

def backward(X, y, p, cache):
    n = X.shape[0]
    A1, A2 = cache["A1"], cache["A2"]
    dZ2 = A2 - y
    dW2 = A1.T @ dZ2 / n
    db2 = np.mean(dZ2, axis=0)
    dA1 = dZ2 @ p["W2"].T
    dZ1 = dA1 * (1 - A1 ** 2)
    dW1 = X.T @ dZ1 / n
    db1 = np.mean(dZ1, axis=0)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
""",
    )

    # ────────────────────────────────────────────────── Wed — XOR
    ex(
        "d10-deep", "Train it on XOR",
        """
## Why XOR is the test

XOR is not linearly separable — no straight line divides the classes. A single
layer *cannot* solve it, and this exact fact is what Minsky and Papert used in
1969 to argue perceptrons were fundamentally limited, collapsing neural network
funding for a generation.

A hidden layer solves it. **If your net learns XOR, your backprop is correct.**
If it stalls at 50% or 75% accuracy, something is wrong.

```
(0,0) -> 0     (0,1) -> 1     (1,0) -> 1     (1,1) -> 0
```

## Your task

`train_xor(steps, lr, h, seed)` returning `(params, losses)`.

You may reuse your `init_params`, `forward`, `backward` — paste them in. Then
the loop is: forward, record loss, backward, update every parameter.
""",
        """
import numpy as np

# paste your init_params / forward / backward here

def train_xor(steps=5000, lr=0.5, h=4, seed=0):
    X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    y = np.array([[0.],[1.],[1.],[0.]])
    pass    # -> (params, losses)

def predict(X, p):
    \"\"\"Hard 0/1 labels, shape (n,1).\"\"\"
    pass
""",
        """
import numpy as np
X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = np.array([[0.],[1.],[1.],[0.]])

p, losses = train_xor(steps=6000, lr=0.5, h=4, seed=0)
check(len(losses) == 6000, f"expected 6000 losses, got {len(losses)}")
check(losses[-1] < losses[0], "loss must decrease")
check(losses[-1] < 0.05, f"final loss should be under 0.05 — got {losses[-1]:.4f}. "
      "Stuck near 0.69 means the net never broke symmetry or lr is too small.")

pred = np.asarray(predict(X, p)).reshape(-1)
check(np.array_equal(pred, np.array([0,1,1,0])),
      f"XOR must be solved exactly: expected [0 1 1 0], got {pred}")

# not just a lucky seed
p2, l2 = train_xor(steps=6000, lr=0.5, h=4, seed=3)
pred2 = np.asarray(predict(X, p2)).reshape(-1)
check(np.array_equal(pred2, np.array([0,1,1,0])),
      "should solve XOR from a different random seed too, not just a lucky one")

# a hidden layer is doing real work: 1 hidden unit should NOT reliably solve it
p3, l3 = train_xor(steps=4000, lr=0.5, h=1, seed=0)
check(l3[-1] > l2[-1],
      "with only 1 hidden unit the loss should stay clearly worse than with 4 — "
      "if it does not, your hidden layer may not be connected properly")
""",
        [
            "Update rule for each parameter: `p['W1'] -= lr * g['dW1']`, and so on for all four.",
            "Record the loss BEFORE the update, so len(losses) == steps.",
            "Loss is binary cross-entropy on A2. Clip before the log.",
            "If the loss sits flat around 0.693, the net is predicting 0.5 for everything — usually a too-small learning rate or zero-initialised weights.",
            "lr=0.5 is large but XOR is tiny and needs it. If it diverges to nan, lower it.",
            "`predict` = `(forward(X, p)[0] >= 0.5).astype(int)`.",
        ],
        """
import numpy as np

def init_params(d, h, seed=0):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0,0.5,(d,h)), "b1": np.zeros(h),
            "W2": rng.normal(0,0.5,(h,1)), "b2": np.zeros(1)}

def forward(X, p):
    Z1 = X @ p["W1"] + p["b1"]; A1 = np.tanh(Z1)
    Z2 = A1 @ p["W2"] + p["b2"]
    A2 = 1/(1+np.exp(-np.clip(Z2,-500,500)))
    return A2, {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}

def backward(X, y, p, cache):
    n = X.shape[0]; A1, A2 = cache["A1"], cache["A2"]
    dZ2 = A2 - y
    dW2 = A1.T @ dZ2 / n; db2 = np.mean(dZ2, axis=0)
    dZ1 = (dZ2 @ p["W2"].T) * (1 - A1**2)
    dW1 = X.T @ dZ1 / n;  db1 = np.mean(dZ1, axis=0)
    return {"dW1":dW1,"db1":db1,"dW2":dW2,"db2":db2}

def train_xor(steps=5000, lr=0.5, h=4, seed=0):
    X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    y = np.array([[0.],[1.],[1.],[0.]])
    p = init_params(2, h, seed); losses = []
    for _ in range(steps):
        A2, cache = forward(X, p)
        pc = np.clip(A2, 1e-12, 1-1e-12)
        losses.append(float(-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc))))
        g = backward(X, y, p, cache)
        for k in ("W1","b1","W2","b2"):
            p[k] = p[k] - lr * g["d"+k]
    return p, losses

def predict(X, p):
    return (forward(X, p)[0] >= 0.5).astype(int)
""",
    )

    # ──────────────────────────────────────── Thu — activation functions
    ex(
        "d11-deep", "Activation functions and their derivatives",
        """
## Why a network needs them at all

Stack two linear layers with no activation between them:

```
(X @ W1) @ W2  =  X @ (W1 @ W2)  =  X @ W
```

Two layers collapse into one. **Without a non-linearity, depth buys you
nothing.** The activation function is the entire reason deep networks can
represent more than a line.

## The three that matter

**sigmoid** — squashes to (0,1). Derivative `σ(1−σ)`, which peaks at 0.25 and
approaches zero at both ends. Saturating like that is what causes vanishing
gradients in deep nets.

**tanh** — squashes to (−1,1), zero-centred, derivative `1 − tanh²` peaking at
1. Strictly better than sigmoid for hidden layers, same saturation problem.

**ReLU** — `max(0, z)`. Derivative is 1 for positive input and 0 otherwise.
Does not saturate on the positive side, which is why it made deep networks
trainable. Its failure mode: a unit stuck at negative input gets zero gradient
forever — a **dead ReLU**.

## Your task

Implement all three plus their derivatives, taking `z` as input. Then
`deriv_max()` returning a dict of the maximum value each derivative reaches.
""",
        """
import numpy as np

def sigmoid(z): pass
def d_sigmoid(z): pass
def tanh(z): pass
def d_tanh(z): pass
def relu(z): pass
def d_relu(z): pass

def deriv_max():
    \"\"\"Return {'sigmoid': ..., 'tanh': ..., 'relu': ...} — the maximum value
    each derivative attains. Think about it; do not brute force.\"\"\"
    pass
""",
        """
import numpy as np
z = np.linspace(-6, 6, 400)

check(abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-9, "sigmoid(0) = 0.5")
check(np.all((sigmoid(z) > 0) & (sigmoid(z) < 1)), "sigmoid must stay in (0,1)")
check(np.all(np.isfinite(sigmoid(np.array([-900.,900.])))), "sigmoid must not overflow")
check(abs(tanh(np.array([0.0]))[0]) < 1e-12, "tanh(0) = 0")
check(np.all(np.abs(tanh(z)) <= 1), "tanh must stay in [-1,1]")
check(np.allclose(relu(np.array([-2.,0.,3.])), [0.,0.,3.]), "relu is max(0,z)")

# derivatives must match numerical differentiation
def numd(f, x, eps=1e-6):
    return (f(np.array([x+eps])) - f(np.array([x-eps])))[0] / (2*eps)
for name, f, df in [("sigmoid",sigmoid,d_sigmoid), ("tanh",tanh,d_tanh)]:
    for x in [-3.0, -0.7, 0.0, 0.7, 3.0]:
        check(abs(float(df(np.array([x]))[0]) - numd(f,x)) < 1e-5,
              f"d_{name} disagrees with numerical derivative at z={x}")
check(np.allclose(d_relu(np.array([-2.,3.,5.])), [0.,1.,1.]),
      "d_relu is 0 for negative, 1 for positive")

m = deriv_max()
check(abs(m["sigmoid"] - 0.25) < 1e-6, f"max sigmoid' is 0.25, got {m['sigmoid']}")
check(abs(m["tanh"] - 1.0) < 1e-6, f"max tanh' is 1.0, got {m['tanh']}")
check(abs(m["relu"] - 1.0) < 1e-6, f"max relu' is 1.0, got {m['relu']}")

# the vanishing-gradient point, made concrete
check(0.25**10 < 1e-6, "0.25^10 is about 1e-6 — this is why deep sigmoid nets stopped learning")
""",
        [
            "`d_sigmoid(z) = sigmoid(z) * (1 - sigmoid(z))`.",
            "`d_tanh(z) = 1 - np.tanh(z)**2`.",
            "`d_relu(z) = (z > 0).astype(float)`.",
            "Clip inside sigmoid to avoid overflow at large |z|.",
            "The maxima: sigmoid' peaks at z=0 giving 0.5*0.5 = 0.25; tanh' peaks at z=0 giving 1; relu' is 1 wherever it is defined and positive.",
        ],
        """
import numpy as np

def sigmoid(z): return 1.0/(1.0+np.exp(-np.clip(z,-500,500)))
def d_sigmoid(z): s = sigmoid(z); return s*(1-s)
def tanh(z): return np.tanh(z)
def d_tanh(z): return 1.0 - np.tanh(z)**2
def relu(z): return np.maximum(0.0, z)
def d_relu(z): return (z > 0).astype(float)
def deriv_max(): return {"sigmoid": 0.25, "tanh": 1.0, "relu": 1.0}
""",
    )

    # ──────────────────────────────── Fri — multi-class + softmax
    ex(
        "d12-deep", "Multi-class: softmax and cross-entropy",
        """
## Note on the data

The scheduled task is MNIST. Do that run in VS Code — it needs a download the
browser cannot do here. **These tests use a synthetic 3-class problem instead**,
which exercises exactly the same machinery: softmax output, categorical
cross-entropy, and the multi-class gradient.

## From two classes to many

Sigmoid gives one probability. For `K` classes you need K probabilities that sum
to 1. That is **softmax**:

```
softmax(z)ₖ = exp(zₖ) / Σⱼ exp(zⱼ)
```

## The overflow trap

`exp(1000)` is infinity. The fix everybody uses: subtract the row maximum first.

```
softmax(z) = exp(z − max(z)) / Σ exp(z − max(z))
```

This changes nothing mathematically — the constant cancels — and it makes the
computation stable. **Every real implementation does this.**

## The gradient, again

With softmax output and categorical cross-entropy, the same cancellation
happens as with sigmoid and BCE:

```
dZ = (A − Y) / n
```

where `Y` is one-hot. Third time you have seen this. That is not coincidence —
it is a property of the exponential family paired with its matching loss.

## Your task

`softmax(Z)` row-wise, `one_hot(y, K)`, `cross_entropy(A, Y)`, and
`train_softmax(X, y, K, h, steps, lr, seed)` returning `(params, losses)` for a
2-layer net with tanh hidden and softmax output.
""",
        """
import numpy as np

def softmax(Z):
    \"\"\"Z:(n,K) -> (n,K), each row summing to 1. Must be overflow-safe.\"\"\"
    pass

def one_hot(y, K):
    \"\"\"y:(n,) of ints -> (n,K) of 0/1.\"\"\"
    pass

def cross_entropy(A, Y):
    \"\"\"Mean categorical cross-entropy. A:(n,K) probs, Y:(n,K) one-hot.\"\"\"
    pass

def train_softmax(X, y, K, h=16, steps=800, lr=0.5, seed=0):
    \"\"\"2-layer net: X -> tanh hidden(h) -> softmax(K). Return (params, losses).\"\"\"
    pass

def predict(X, p):
    \"\"\"Return predicted class indices, shape (n,).\"\"\"
    pass
""",
        """
import numpy as np

Z = np.array([[1.,2.,3.],[1000.,1000.,1000.]])
S = softmax(Z)
check(np.all(np.isfinite(S)), "softmax must not overflow at z=1000 — subtract the row max")
check(np.allclose(S.sum(axis=1), 1.0), "each row of softmax must sum to 1")
check(np.allclose(S[1], [1/3,1/3,1/3]), "equal logits should give equal probabilities")
check(S[0,2] > S[0,1] > S[0,0], "larger logit must give larger probability")
check(np.allclose(softmax(Z), softmax(Z + 7.0)),
      "softmax must be shift-invariant — adding a constant changes nothing")

Y = one_hot(np.array([0,2,1]), 3)
check(np.array_equal(Y, [[1,0,0],[0,0,1],[0,1,0]]), f"one_hot wrong: {Y}")

perfect = np.array([[1-1e-9,5e-10,5e-10]])
check(cross_entropy(perfect, np.array([[1.,0.,0.]])) < 1e-6,
      "a perfect confident prediction should have ~zero loss")
check(cross_entropy(np.full((1,3),1/3), np.array([[1.,0.,0.]])) > 1.0,
      "a uniform prediction over 3 classes should cost about ln(3)=1.10")

# three well-separated blobs
rng = np.random.default_rng(21)
cent = np.array([[-4.,-4.],[4.,-4.],[0.,5.]])
X = np.vstack([rng.normal(c, 0.7, (80,2)) for c in cent])
y = np.repeat([0,1,2], 80)

p, losses = train_softmax(X, y, K=3, h=16, steps=800, lr=0.5, seed=0)
check(len(losses) == 800, f"expected 800 losses, got {len(losses)}")
check(losses[-1] < losses[0], "loss must decrease")
acc = (np.asarray(predict(X, p)).reshape(-1) == y).mean()
check(acc > 0.95, f"accuracy on separable blobs should exceed 95%, got {acc:.3f}")

Xh = np.vstack([rng.normal(c, 0.7, (30,2)) for c in cent])
yh = np.repeat([0,1,2], 30)
check((np.asarray(predict(Xh,p)).reshape(-1) == yh).mean() > 0.93, "must generalise")
""",
        [
            "`Zs = Z - Z.max(axis=1, keepdims=True)` then `e = np.exp(Zs)` then divide by `e.sum(axis=1, keepdims=True)`.",
            "`one_hot`: `np.eye(K)[y]` is the one-liner.",
            "Cross-entropy: `-np.mean(np.sum(Y * np.log(np.clip(A,1e-12,1)), axis=1))`.",
            "Forward: `Z1 = X@W1+b1; A1 = tanh(Z1); Z2 = A1@W2+b2; A2 = softmax(Z2)`.",
            "Backward: `dZ2 = (A2 - Y)/n`, then exactly the same chain as the binary case.",
            "W2 now has shape (h, K) and b2 has shape (K,).",
            "`predict` = `np.argmax(forward(X,p)[0], axis=1)`.",
        ],
        """
import numpy as np

def softmax(Z):
    e = np.exp(Z - Z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def one_hot(y, K):
    return np.eye(K)[np.asarray(y).astype(int)]

def cross_entropy(A, Y):
    return float(-np.mean(np.sum(Y * np.log(np.clip(A,1e-12,1.0)), axis=1)))

def _fwd(X, p):
    Z1 = X @ p["W1"] + p["b1"]; A1 = np.tanh(Z1)
    Z2 = A1 @ p["W2"] + p["b2"]; A2 = softmax(Z2)
    return A2, {"A1":A1, "A2":A2}

def train_softmax(X, y, K, h=16, steps=800, lr=0.5, seed=0):
    rng = np.random.default_rng(seed); n, d = X.shape
    p = {"W1":rng.normal(0,0.3,(d,h)), "b1":np.zeros(h),
         "W2":rng.normal(0,0.3,(h,K)), "b2":np.zeros(K)}
    Y = one_hot(y, K); losses = []
    for _ in range(steps):
        A2, c = _fwd(X, p)
        losses.append(cross_entropy(A2, Y))
        dZ2 = (A2 - Y) / n
        dW2 = c["A1"].T @ dZ2; db2 = dZ2.sum(axis=0)
        dZ1 = (dZ2 @ p["W2"].T) * (1 - c["A1"]**2)
        dW1 = X.T @ dZ1;       db1 = dZ1.sum(axis=0)
        p["W2"] -= lr*dW2; p["b2"] -= lr*db2
        p["W1"] -= lr*dW1; p["b1"] -= lr*db1
    return p, losses

def predict(X, p):
    return np.argmax(_fwd(X, p)[0], axis=1)
""",
    )

    # ────────────────────────────────────────── Sat — week 2 rebuild
    ex(
        "d13-deep", "Week-2 rebuild — from a blank file",
        """
## From memory. Close everything else.

Rebuild a working 2-layer binary classifier:

1. `init_params(d, h, seed)` — small random W, zero b
2. `forward(X, p)` — tanh hidden, sigmoid output, returns `(A2, cache)`
3. `backward(X, y, p, cache)` — all four gradients
4. `train(X, y, h, steps, lr, seed)` — returns `(params, losses)`
5. `predict(X, p)` — hard 0/1 labels

The tests include **numerical gradient checking**, so a backprop that is
approximately right will fail. It has to be exact.

If you cannot derive the gradients, do not open the solution — derive them on
paper again. That derivation is the entire point of this week.
""",
        """
import numpy as np

def init_params(d, h, seed=0): pass
def forward(X, p): pass
def backward(X, y, p, cache): pass
def train(X, y, h=8, steps=3000, lr=0.5, seed=0): pass
def predict(X, p): pass
""",
        """
import numpy as np
rng = np.random.default_rng(31)

p = init_params(2, 5, seed=0)
check(np.asarray(p["W1"]).shape==(2,5) and np.asarray(p["W2"]).shape==(5,1),
      "parameter shapes are wrong")
check(not np.allclose(p["W1"], 0), "weights must not start at zero")

X = rng.normal(0,1,(20,2)); y = (rng.random((20,1))>0.5).astype(float)
A2, cache = forward(X, p)
check(np.asarray(A2).shape==(20,1) and np.all((A2>0)&(A2<1)),
      "forward must return (n,1) probabilities")

def _loss(pp):
    a = forward(X, pp)[0]; a = np.clip(a,1e-12,1-1e-12)
    return float(-np.mean(y*np.log(a)+(1-y)*np.log(1-a)))
g = backward(X, y, p, cache)
bad = []
for key, gk in [("W1","dW1"),("b1","db1"),("W2","dW2"),("b2","db2")]:
    for i in range(np.asarray(p[key]).size):
        pp = {k:v.copy() for k,v in p.items()}
        pp[key].flat[i] += 1e-6; a = _loss(pp)
        pp[key].flat[i] -= 2e-6; b = _loss(pp)
        num = (a-b)/2e-6; ana = float(np.asarray(g[gk]).flat[i])
        if abs(num-ana) > 1e-5*max(1.0,abs(num)): bad.append(f"{gk}[{i}]")
check(not bad, f"gradient check failed on: {bad[:4]}")

# XOR: the real proof
Xx = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]); yx = np.array([[0.],[1.],[1.],[0.]])
pp, losses = train(Xx, yx, h=4, steps=6000, lr=0.5, seed=0)
check(losses[-1] < 0.05, f"XOR loss should end under 0.05, got {losses[-1]:.4f}")
check(np.array_equal(np.asarray(predict(Xx,pp)).reshape(-1), [0,1,1,0]),
      "must solve XOR exactly")

# and a harder, noisier problem
Xc = rng.normal(0,1,(300,2))
yc = ((Xc[:,0]**2 + Xc[:,1]**2) > 1.5).astype(float).reshape(-1,1)
pc, lc = train(Xc, yc, h=12, steps=4000, lr=0.5, seed=1)
acc = (np.asarray(predict(Xc,pc)).reshape(-1) == yc.reshape(-1)).mean()
check(acc > 0.90, f"should learn a circular boundary to >90%, got {acc:.3f}")
""",
        [
            "Write the forward pass first and check shapes before touching backward.",
            "dZ2 = A2 - y. Everything follows from there.",
            "The pattern each layer: dZ from the incoming gradient times the local derivative; dW = input.T @ dZ / n; pass back dZ @ W.T.",
            "If the gradient check fails only on W1, the tanh derivative or the dA1 step is wrong.",
        ],
        """
import numpy as np

def init_params(d, h, seed=0):
    rng = np.random.default_rng(seed)
    return {"W1":rng.normal(0,0.5,(d,h)), "b1":np.zeros(h),
            "W2":rng.normal(0,0.5,(h,1)), "b2":np.zeros(1)}

def forward(X, p):
    Z1 = X @ p["W1"] + p["b1"]; A1 = np.tanh(Z1)
    Z2 = A1 @ p["W2"] + p["b2"]
    A2 = 1/(1+np.exp(-np.clip(Z2,-500,500)))
    return A2, {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}

def backward(X, y, p, cache):
    n = X.shape[0]; A1, A2 = cache["A1"], cache["A2"]
    dZ2 = A2 - y
    dW2 = A1.T @ dZ2 / n; db2 = np.mean(dZ2, axis=0)
    dZ1 = (dZ2 @ p["W2"].T) * (1 - A1**2)
    dW1 = X.T @ dZ1 / n;  db1 = np.mean(dZ1, axis=0)
    return {"dW1":dW1,"db1":db1,"dW2":dW2,"db2":db2}

def train(X, y, h=8, steps=3000, lr=0.5, seed=0):
    p = init_params(X.shape[1], h, seed); losses=[]
    for _ in range(steps):
        A2, cache = forward(X, p)
        a = np.clip(A2,1e-12,1-1e-12)
        losses.append(float(-np.mean(y*np.log(a)+(1-y)*np.log(1-a))))
        g = backward(X, y, p, cache)
        for k in ("W1","b1","W2","b2"): p[k] = p[k] - lr*g["d"+k]
    return p, losses

def predict(X, p):
    return (forward(X, p)[0] >= 0.5).astype(int)
""",
    )
