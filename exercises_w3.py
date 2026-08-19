"""Week 3 — build an autograd engine. Imported by build_exercises.py."""


def add(ex):
    # ─────────────────────────────────────── Mon — the Value class, forward
    ex(
        "d15-deep", "micrograd pt.1 — the Value class and the graph",
        """
## What you are building this week

PyTorch in about 100 lines. By Saturday you will have written an automatic
differentiation engine and trained a neural network on it.

## The idea

Wrap every number in an object that remembers **where it came from**.

```
a = Value(2.0)
b = Value(-3.0)
c = a * b        # c knows: I am a product, and my parents are a and b
```

Do that for every operation and you get a **graph** of the whole computation.
Then differentiating is walking that graph backwards.

That is the entire trick behind every deep learning framework.

## Today: forward only

Build `Value` with:

- `.data` — the number
- `.grad` — the derivative, initialised to 0.0 (unused today)
- `._prev` — a **set** of the Values that produced this one
- `._op` — a string naming the operation, for debugging

Support `+` and `*` between two Values, and between a Value and a plain number.

## The Python detail that trips people

`2 * a` calls `int.__mul__(2, a)` first, which fails, and Python then tries
`a.__rmul__(2)`. So you need `__radd__` and `__rmul__` or `2 * a` raises a
TypeError while `a * 2` works fine.
""",
        """
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass

    __radd__ = None   # replace
    __rmul__ = None   # replace
""",
        """
a = Value(2.0); b = Value(-3.0)
c = a * b
check(abs(c.data + 6.0) < 1e-12, f"2 * -3 should be -6, got {c.data}")
check(c._prev == {a, b}, "c._prev must contain exactly its two parents")
check(c._op == '*', f"c._op should be '*', got {c._op!r}")

d = a + b
check(abs(d.data - (-1.0)) < 1e-12, f"2 + -3 should be -1, got {d.data}")
check(d._op == '+', "the op label for addition should be '+'")
check(a.grad == 0.0, "grad must initialise to 0.0")

# plain numbers on the right
e = a * 3
check(abs(e.data - 6.0) < 1e-12, "Value * number must work")
check(len(e._prev) == 2, "the number should be wrapped in a Value and become a parent")
f = a + 10
check(abs(f.data - 12.0) < 1e-12, "Value + number must work")

# plain numbers on the LEFT — needs __radd__ / __rmul__
g = 3 * a
check(abs(g.data - 6.0) < 1e-12, "number * Value must work — you need __rmul__")
h = 10 + a
check(abs(h.data - 12.0) < 1e-12, "number + Value must work — you need __radd__")

# a real expression graph
x1, x2 = Value(2.0), Value(0.0)
w1, w2 = Value(-3.0), Value(1.0)
n = x1*w1 + x2*w2 + Value(6.881373587019543)
check(abs(n.data - 0.881373587019543) < 1e-9, f"expression value wrong: {n.data}")
check(len(n._prev) == 2, "the final node should have two parents")

# Values must be hashable to live in a set
check(len({Value(1.0), Value(1.0)}) == 2,
      "two distinct Value objects must be distinct set members (do not define __eq__)")
""",
        [
            "In `__add__`, first do `other = other if isinstance(other, Value) else Value(other)`.",
            "Return `Value(self.data + other.data, (self, other), '+')`.",
            "`__mul__` is the same shape with `*`.",
            "`__radd__ = __add__` and `__rmul__ = __mul__` — assign them after defining the methods, inside the class body.",
            "Do not define `__eq__` or `__hash__`. The default identity-based versions are exactly what the set needs.",
        ],
        """
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), '*')

    __radd__ = __add__
    __rmul__ = __mul__
""",
    )

    # ────────────────────────────────────────── Tue — backward
    ex(
        "d16-deep", "micrograd pt.2 — backward and topological sort",
        """
## Today is the day

You implement automatic differentiation. This is the centre of the entire
programme.

## Local derivatives

Each operation knows how to pass gradient to its inputs. That is all it needs to
know — nothing about the rest of the graph.

**Addition** distributes gradient unchanged. If `c = a + b` then `∂c/∂a = 1`,
so `a.grad += c.grad`.

**Multiplication** swaps. If `c = a * b` then `∂c/∂a = b`, so
`a.grad += b.data * c.grad`.

**tanh**: `∂/∂x tanh(x) = 1 − tanh(x)²`, so `x.grad += (1 - t**2) * out.grad`.

## Why `+=` and never `=`

If a Value is used twice — `b = a + a` — gradient arrives from two paths and
they must **accumulate**. Assigning instead of adding silently discards one
path. This is the single most common bug in hand-written autograd, and it gives
wrong answers rather than errors.

## Why topological order

You cannot compute a node's contribution until its own gradient is final. So
sort the graph so every node comes after everything that depends on it, then
walk it in reverse.

Seed the output with `grad = 1.0` — the derivative of anything with respect to
itself.

## Your task

Add `_backward` closures to `+`, `*` and `tanh`, plus a `backward()` method
that topologically sorts and walks the graph in reverse.
""",
        """
import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            pass
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            pass
        out._backward = _backward
        return out

    def tanh(self):
        pass

    def backward(self):
        \"\"\"Topologically sort, seed self.grad = 1.0, walk in reverse.\"\"\"
        pass

    __radd__ = __add__
    __rmul__ = __mul__
""",
        """
import math

# the classic worked example
x1, x2 = Value(2.0), Value(0.0)
w1, w2 = Value(-3.0), Value(1.0)
b = Value(6.8813735870195432)
n = x1*w1 + x2*w2 + b
o = n.tanh()
check(abs(o.data - 0.7071067811865476) < 1e-9, f"tanh output wrong: {o.data}")

o.backward()
check(abs(o.grad - 1.0) < 1e-12, "the output node's grad must be seeded to 1.0")
check(abs(n.grad - 0.5) < 1e-6, f"dn should be 0.5, got {n.grad}")
check(abs(x1.grad - (-1.5)) < 1e-6, f"x1.grad should be -1.5, got {x1.grad}")
check(abs(w1.grad - 1.0) < 1e-6, f"w1.grad should be 1.0, got {w1.grad}")
check(abs(x2.grad - 0.5) < 1e-6, f"x2.grad should be 0.5, got {x2.grad}")
check(abs(w2.grad - 0.0) < 1e-6, f"w2.grad should be 0.0, got {w2.grad}")

# ACCUMULATION: a used twice
a = Value(3.0)
y = a + a
y.backward()
check(abs(a.grad - 2.0) < 1e-9,
      f"a used twice must accumulate to 2.0, got {a.grad} — you assigned instead of +=")

a2 = Value(3.0)
y2 = a2 * a2
y2.backward()
check(abs(a2.grad - 6.0) < 1e-9, f"d(a*a)/da at a=3 is 6, got {a2.grad}")

# deeper chain, checked numerically
def f(t):
    v = Value(t)
    return ((v * 2 + 1) * v).tanh()
p = Value(0.7)
out = ((p * 2 + 1) * p).tanh()
out.backward()
eps = 1e-6
num = (f(0.7+eps).data - f(0.7-eps).data) / (2*eps)
check(abs(p.grad - num) < 1e-5,
      f"chained expression gradient {p.grad:.8f} disagrees with numerical {num:.8f}")

# topological order must be respected
q = Value(2.0)
r = q * 3
s = r + q          # q feeds both r and s
t = s * r
t.backward()
numq = None
def g(tv):
    v = Value(tv); rr = v*3; ss = rr+v; return (ss*rr).data
numq = (g(2.0+eps) - g(2.0-eps))/(2*eps)
check(abs(q.grad - numq) < 1e-4,
      f"diamond-shaped graph gradient wrong: {q.grad:.6f} vs numerical {numq:.6f}")
""",
        [
            "Addition: `def _backward(): self.grad += out.grad; other.grad += out.grad`.",
            "Multiplication: `self.grad += other.data * out.grad` and `other.grad += self.data * out.grad`.",
            "tanh: `t = math.tanh(self.data)`, `out = Value(t, (self,), 'tanh')`, and `self.grad += (1 - t**2) * out.grad`.",
            "The closure captures `out` — that is why it must be defined after `out` exists.",
            "Topological sort: a recursive `build(v)` that visits every child before appending v to a list, guarded by a `visited` set.",
            "Then `self.grad = 1.0` and `for node in reversed(topo): node._backward()`.",
        ],
        """
import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data); self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children); self._op = _op

    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo): node._backward()

    __radd__ = __add__
    __rmul__ = __mul__
""",
    )

    # ─────────────────────────────────────── Wed — an MLP on your engine
    ex(
        "d17-deep", "micrograd pt.3 — a neural network on your engine",
        """
## Now use it

You have autograd. Build a network on top and train it — and notice that you
never write a single derivative again. The engine handles all of it.

## The three classes

**Neuron** — `nin` weights plus a bias, all Values. Calling it computes
`tanh(Σ wᵢxᵢ + b)`.

**Layer** — a list of Neurons. Calling it returns a list of outputs (or the bare
Value if there is only one).

**MLP** — a list of Layers, applied in sequence. `MLP(3, [4, 4, 1])` means 3
inputs, two hidden layers of 4, one output.

**`parameters()`** on each returns every Value that should be trained. This is
what the optimiser walks.

## The training loop

```
for step in range(steps):
    preds = [model(x) for x in xs]
    loss  = sum((p - t)**2 for p, t in zip(preds, ys))

    for p in model.parameters(): p.grad = 0.0    # <- do not forget
    loss.backward()

    for p in model.parameters(): p.data -= lr * p.grad
```

## The bug everyone hits

**Zero the gradients before every backward pass.** Gradients accumulate by
design — that is what makes reuse work — so if you forget, step 100 is using
the sum of all previous gradients and training silently diverges. PyTorch's
`optimizer.zero_grad()` exists for exactly this reason.

## Your task

Paste your Value class from yesterday, add `__pow__`, `__neg__` and `__sub__`
(you need `(pred - target)**2`), then build Neuron, Layer, MLP and `train`.
""",
        """
import math, random

# paste your Value class here, then add:
#   __pow__(self, k)  for integer/float k
#   __neg__, __sub__

class Neuron:
    def __init__(self, nin, rng): pass
    def __call__(self, x): pass
    def parameters(self): pass

class Layer:
    def __init__(self, nin, nout, rng): pass
    def __call__(self, x): pass
    def parameters(self): pass

class MLP:
    def __init__(self, nin, nouts, seed=0): pass
    def __call__(self, x): pass
    def parameters(self): pass

def train(model, xs, ys, steps=100, lr=0.05):
    \"\"\"Squared-error loss. Return the list of loss values (floats).\"\"\"
    pass
""",
        """
import math

# __pow__ / __sub__ must work and differentiate correctly
a = Value(3.0)
c = a ** 2
c.backward()
check(abs(c.data - 9.0) < 1e-9, "3**2 should be 9")
check(abs(a.grad - 6.0) < 1e-9, f"d(a^2)/da at 3 is 6, got {a.grad}")

d = Value(5.0) - Value(2.0)
check(abs(d.data - 3.0) < 1e-9, "subtraction should work")

m = MLP(3, [4, 4, 1], seed=0)
ps = m.parameters()
check(len(ps) == (3*4+4) + (4*4+4) + (4*1+1),
      f"MLP(3,[4,4,1]) should have 41 parameters, got {len(ps)}")
check(all(isinstance(p, Value) for p in ps), "parameters() must return Value objects")

out = m([2.0, 3.0, -1.0])
check(isinstance(out, Value), "a single-output MLP should return a bare Value")
check(-1.0 <= out.data <= 1.0, "tanh output must lie in [-1,1]")

# the classic 4-example dataset
xs = [[2.0,3.0,-1.0],[3.0,-1.0,0.5],[0.5,1.0,1.0],[1.0,1.0,-1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1], seed=0)
losses = train(model, xs, ys, steps=200, lr=0.05)
check(len(losses) == 200, f"expected 200 losses, got {len(losses)}")
check(losses[-1] < losses[0], "loss must decrease")
check(losses[-1] < 0.01, f"loss should fall below 0.01 in 200 steps, got {losses[-1]:.5f}")

preds = [model(x).data for x in xs]
check(all((p > 0) == (t > 0) for p, t in zip(preds, ys)),
      f"every prediction should have the right sign: {[round(p,3) for p in preds]}")

# gradients must be zeroed each step
m2 = MLP(2, [3, 1], seed=1)
l2 = train(m2, [[1.0,2.0],[-1.0,0.5]], [1.0,-1.0], steps=150, lr=0.05)
check(l2[-1] < l2[0] and l2[-1] < 0.05,
      "if the loss stalls or explodes you are probably not zeroing grads each step")
check(all(abs(p.grad) < 1e3 for p in m2.parameters()),
      "gradients look accumulated across steps — zero them before each backward()")
""",
        [
            "`__pow__`: `out = Value(self.data ** k, (self,), f'**{k}')` with `self.grad += k * self.data**(k-1) * out.grad`.",
            "`__neg__` is `self * -1`; `__sub__` is `self + (-other)`.",
            "Neuron: `self.w = [Value(rng.uniform(-1,1)) for _ in range(nin)]`, `self.b = Value(0.0)`.",
            "Calling a Neuron: `act = sum((wi*xi for wi,xi in zip(self.w,x)), self.b)` then `act.tanh()`.",
            "Layer parameters: `[p for n in self.neurons for p in n.parameters()]`.",
            "MLP: `sz = [nin] + nouts`, build `Layer(sz[i], sz[i+1])` for each i.",
            "In train: zero every parameter's grad, then loss.backward(), then update.",
        ],
        """
import math, random

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data); self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children); self._op = _op
    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _b(): self.grad += out.grad; other.grad += out.grad
        out._backward = _b; return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _b():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _b; return out
    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f'**{k}')
        def _b(): self.grad += k * (self.data ** (k-1)) * out.grad
        out._backward = _b; return out
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-(other if isinstance(other, Value) else Value(other)))
    def __rsub__(self, other): return Value(other) + (-self)
    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,), 'tanh')
        def _b(): self.grad += (1 - t**2) * out.grad
        out._backward = _b; return out
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self); self.grad = 1.0
        for n in reversed(topo): n._backward()
    __radd__ = __add__
    __rmul__ = __mul__

class Neuron:
    def __init__(self, nin, rng):
        self.w = [Value(rng.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0.0)
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w,x)), self.b)
        return act.tanh()
    def parameters(self): return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout, rng):
        self.neurons = [Neuron(nin, rng) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self): return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts, seed=0):
        rng = random.Random(seed)
        sz = [nin] + list(nouts)
        self.layers = [Layer(sz[i], sz[i+1], rng) for i in range(len(nouts))]
    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x
    def parameters(self): return [p for l in self.layers for p in l.parameters()]

def train(model, xs, ys, steps=100, lr=0.05):
    losses = []
    for _ in range(steps):
        preds = [model(x) for x in xs]
        loss = sum(((p - t)**2 for p, t in zip(preds, ys)), Value(0.0))
        losses.append(loss.data)
        for p in model.parameters(): p.grad = 0.0
        loss.backward()
        for p in model.parameters(): p.data -= lr * p.grad
    return losses
""",
    )

    # ────────────────────────────────────────── Thu — extend the engine
    ex(
        "d18-deep", "Extend micrograd — exp, log, ReLU, division",
        """
## Not from the video

Karpathy's version has tanh. You are adding four more operations yourself, from
the derivatives, with no reference to follow. This is the difference between
having watched a lecture and having understood it.

| operation | derivative | passes back |
|---|---|---|
| `exp(x)` | `eˣ` | `out.data * out.grad` |
| `log(x)` | `1/x` | `(1/x) * out.grad` |
| `relu(x)` | 1 if x>0 else 0 | `(out.data > 0) * out.grad` |
| `a / b` | — | implement as `a * b**-1` |

## The one that is nearly free

Division needs no new backward code at all — if `__pow__` handles negative
exponents, then `a / b` is `a * b**(-1)` and the existing rules cover it.

**Reusing primitives instead of adding special cases is how real frameworks stay
small.** PyTorch does the same thing.

## Note on relu's derivative at zero

It is undefined — there is a kink. Every framework picks a convention; use 0.
It essentially never matters in practice, and knowing that it is a convention
rather than a fact is the point.

## Your task

Paste your Value class, add `exp`, `log`, `relu`, `__truediv__` and
`__rtruediv__`, all differentiating correctly.
""",
        """
import math

# paste your Value class, then add exp / log / relu / __truediv__ / __rtruediv__
""",
        """
import math
eps = 1e-6
def numgrad(fn, x0):
    return (fn(x0+eps) - fn(x0-eps)) / (2*eps)

# exp
a = Value(0.7); e = a.exp(); e.backward()
check(abs(e.data - math.exp(0.7)) < 1e-9, "exp value wrong")
check(abs(a.grad - math.exp(0.7)) < 1e-6, f"d(exp)/dx = exp(x); got {a.grad}")

# log
b = Value(2.5); l = b.log(); l.backward()
check(abs(l.data - math.log(2.5)) < 1e-9, "log value wrong")
check(abs(b.grad - 1/2.5) < 1e-6, f"d(log)/dx = 1/x = 0.4; got {b.grad}")

# relu, both sides
p = Value(3.0); rp = p.relu(); rp.backward()
check(abs(rp.data - 3.0) < 1e-12 and abs(p.grad - 1.0) < 1e-12, "relu positive side wrong")
n = Value(-3.0); rn = n.relu(); rn.backward()
check(abs(rn.data) < 1e-12 and abs(n.grad) < 1e-12, "relu negative side must give 0 value and 0 grad")

# division
c, d = Value(6.0), Value(3.0)
q = c / d; q.backward()
check(abs(q.data - 2.0) < 1e-9, "6/3 should be 2")
check(abs(c.grad - 1/3) < 1e-6, f"d(c/d)/dc = 1/d = 0.333; got {c.grad}")
check(abs(d.grad + 6/9) < 1e-6, f"d(c/d)/dd = -c/d^2 = -0.667; got {d.grad}")

r = 12 / Value(4.0)
check(abs(r.data - 3.0) < 1e-9, "number / Value must work — you need __rtruediv__")

# a composite expression checked numerically
def f(t):
    v = Value(t)
    return (((v*2).exp() + Value(1.0)).log() / (v + Value(3.0))).data
z = Value(0.4)
out = (((z*2).exp() + Value(1.0)).log() / (z + Value(3.0)))
out.backward()
check(abs(z.grad - numgrad(f, 0.4)) < 1e-4,
      f"composite gradient {z.grad:.8f} vs numerical {numgrad(f,0.4):.8f}")

# sigmoid built from your primitives
def sigmoid(v): return 1 / (1 + (-v).exp())
s_in = Value(0.3); s = sigmoid(s_in); s.backward()
expect = 1/(1+math.exp(-0.3))
check(abs(s.data - expect) < 1e-9, "sigmoid built from exp/div is wrong")
check(abs(s_in.grad - expect*(1-expect)) < 1e-6,
      "sigmoid gradient should be s(1-s) and follow automatically from your primitives")
""",
        [
            "`exp`: `out = Value(math.exp(self.data), (self,), 'exp')` and `self.grad += out.data * out.grad`.",
            "`log`: `self.grad += (1/self.data) * out.grad`.",
            "`relu`: value is `max(0, self.data)`; grad is `(out.data > 0) * out.grad`.",
            "`__truediv__(self, other)`: `return self * other**-1`.",
            "For that to work, `__pow__` must accept negative exponents — it already does if you wrote `self.data ** k`.",
            "`__rtruediv__(self, other)`: `return Value(other) * self**-1`.",
        ],
        """
import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data); self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children); self._op = _op
    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _b(): self.grad += out.grad; other.grad += out.grad
        out._backward = _b; return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _b():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _b; return out
    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f'**{k}')
        def _b(): self.grad += k * (self.data ** (k-1)) * out.grad
        out._backward = _b; return out
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-(other if isinstance(other,Value) else Value(other)))
    def __rsub__(self, other): return Value(other) + (-self)
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1
    def __rtruediv__(self, other): return Value(other) * self**-1
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _b(): self.grad += out.data * out.grad
        out._backward = _b; return out
    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _b(): self.grad += (1.0/self.data) * out.grad
        out._backward = _b; return out
    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'relu')
        def _b(): self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _b; return out
    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,), 'tanh')
        def _b(): self.grad += (1 - t**2) * out.grad
        out._backward = _b; return out
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self); self.grad = 1.0
        for n in reversed(topo): n._backward()
    __radd__ = __add__
    __rmul__ = __mul__
""",
    )

    # ─────────────────────────────────────── Fri — gradient checking
    ex(
        "d19-deep", "Gradient checking — proving your engine is right",
        """
## The skill nobody teaches

You have written an autograd engine. How do you *know* it is correct?

The answer is a real research technique, and it is the reason you can trust any
hand-derived gradient: compare against the definition of a derivative.

```
                f(x + ε) − f(x − ε)
    f'(x)  ≈   ─────────────────────
                        2ε
```

This is the **central difference**. It is accurate to O(ε²), while the naive
one-sided version is only O(ε) — for the same cost of two evaluations, it is far
better. Always use the central form.

## Choosing ε

Too large and you measure the curve, not the tangent. Too small and floating
point cancellation destroys you — subtracting two nearly equal numbers loses
precision catastrophically. `1e-5` to `1e-7` is the useful window.

## Compare relatively, not absolutely

An absolute difference of 0.001 is fine for a gradient of 5000 and a disaster
for a gradient of 0.0001. Use **relative error**:

```
        |analytic − numerical|
rel =  ─────────────────────────
        max(|analytic|, |numerical|, tiny)
```

Under 1e-7 is excellent. Under 1e-5 is fine. Above 1e-3, you have a bug.

## Your task

`numerical_grad(f, x, eps)`, `rel_error(a, b)`, and `grad_check(expr_fn, xs)`
which builds the expression from Values, runs your `backward()`, and returns the
worst relative error across all inputs.
""",
        """
import math

# paste your Value class here

def numerical_grad(f, x, eps=1e-6):
    \"\"\"Central difference derivative of scalar f at scalar x.\"\"\"
    pass

def rel_error(a, b):
    \"\"\"Relative error between two numbers, safe when both are ~0.\"\"\"
    pass

def grad_check(expr_fn, xs, eps=1e-6):
    \"\"\"expr_fn takes a list of Values and returns a single Value.
    xs is a list of floats.
    Build Values, call expr_fn, backward(), and compare every input's .grad
    against the numerical derivative. Return the WORST relative error.\"\"\"
    pass
""",
        """
import math

check(abs(numerical_grad(lambda x: x**2, 3.0) - 6.0) < 1e-5, "d(x^2)/dx at 3 is 6")
check(abs(numerical_grad(math.sin, 0.0) - 1.0) < 1e-5, "d(sin)/dx at 0 is 1")
check(abs(numerical_grad(math.exp, 1.0) - math.e) < 1e-4, "d(exp)/dx at 1 is e")

check(rel_error(1.0, 1.0) < 1e-12, "identical values have zero relative error")
check(rel_error(0.0, 0.0) < 1e-12, "rel_error(0,0) must not divide by zero")
check(abs(rel_error(1.0, 1.1) - 0.1/1.1) < 1e-6 or abs(rel_error(1.0,1.1) - 0.1) < 0.02,
      "rel_error should scale by magnitude")
check(rel_error(1e6, 1e6 + 1.0) < 1e-5, "a difference of 1 in a million is small relatively")

# your engine must survive these
worst = grad_check(lambda v: v[0]*v[1] + v[0], [2.0, -3.0])
check(worst < 1e-5, f"simple product+sum failed grad check: {worst:.2e}")

worst = grad_check(lambda v: (v[0]*v[1] + v[2]).tanh(), [2.0, -3.0, 1.0])
check(worst < 1e-5, f"tanh expression failed grad check: {worst:.2e}")

worst = grad_check(lambda v: v[0]*v[0]*v[0] + v[0]*v[1], [1.5, 2.0])
check(worst < 1e-5, f"repeated-use expression failed: {worst:.2e} "
      "(this is the accumulation test — a += bug shows up here)")

worst = grad_check(lambda v: ((v[0].exp() + v[1]) / (v[0] + 3.0)).log(), [0.5, 2.0])
check(worst < 1e-4, f"composite exp/div/log expression failed: {worst:.2e}")

# a deep chain
def deep(v):
    x = v[0]
    for _ in range(8): x = (x * 0.9 + 0.1).tanh()
    return x
check(grad_check(deep, [0.3]) < 1e-4, "an 8-deep chain should still check out")
""",
        [
            "`numerical_grad`: `(f(x+eps) - f(x-eps)) / (2*eps)`.",
            "`rel_error`: `abs(a-b) / max(abs(a), abs(b), 1e-12)`.",
            "In grad_check, build `vals = [Value(x) for x in xs]`, call `out = expr_fn(vals)`, then `out.backward()`.",
            "For the numerical side, define a helper that rebuilds the whole expression with one input nudged: `expr_fn([Value(x) for x in perturbed]).data`.",
            "Rebuild the Values each time — a Value already carries a grad from the last pass.",
            "Return `max(...)` over the per-input relative errors.",
        ],
        """
import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data); self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children); self._op = _op
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _b(): self.grad += out.grad; other.grad += out.grad
        out._backward = _b; return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _b():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _b; return out
    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f'**{k}')
        def _b(): self.grad += k * (self.data ** (k-1)) * out.grad
        out._backward = _b; return out
    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-(o if isinstance(o,Value) else Value(o)))
    def __truediv__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return self * o**-1
    def __rtruediv__(self, o): return Value(o) * self**-1
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _b(): self.grad += out.data * out.grad
        out._backward = _b; return out
    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _b(): self.grad += (1.0/self.data) * out.grad
        out._backward = _b; return out
    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,), 'tanh')
        def _b(): self.grad += (1-t**2) * out.grad
        out._backward = _b; return out
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self); self.grad = 1.0
        for n in reversed(topo): n._backward()
    __radd__ = __add__
    __rmul__ = __mul__

def numerical_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

def rel_error(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-12)

def grad_check(expr_fn, xs, eps=1e-6):
    vals = [Value(x) for x in xs]
    out = expr_fn(vals)
    out.backward()
    worst = 0.0
    for i in range(len(xs)):
        def f(t, i=i):
            pert = list(xs); pert[i] = t
            return expr_fn([Value(v) for v in pert]).data
        num = numerical_grad(f, xs[i], eps)
        worst = max(worst, rel_error(vals[i].grad, num))
    return worst
""",
    )

    # ─────────────────────────────────────── Sat — rebuild micrograd
    ex(
        "d20-deep", "Week-3 rebuild — micrograd from a blank file",
        """
## The most valuable rebuild in the programme

From memory, no references. A complete autograd engine:

- `Value` with `.data`, `.grad`, `._prev`, `._backward`
- `+`, `*`, `**`, unary `-`, `-`, `/` and the reflected versions
- `tanh`, `exp`, `relu`
- `backward()` with topological sort and gradient accumulation

Aim for under 150 lines.

## Why this one matters most

If you can write this from nothing, you understand automatic differentiation —
which means you understand what PyTorch is doing when you call `.backward()`,
and you will never again treat it as magic.

It is also a genuinely good thing to be able to say in an interview, and a
five-minute whiteboard answer that very few candidates can give.

## The tests

Gradient checking against numerical differentiation on several expressions,
including repeated-use graphs that catch the accumulation bug.
""",
        """
import math

# Blank file. From memory.

class Value:
    pass
""",
        """
import math

def numgrad(f, x, eps=1e-6): return (f(x+eps) - f(x-eps)) / (2*eps)
def rel(a, b): return abs(a-b) / max(abs(a), abs(b), 1e-12)

# basics
a, b = Value(2.0), Value(-3.0)
check(abs((a+b).data + 1.0) < 1e-12, "addition wrong")
check(abs((a*b).data + 6.0) < 1e-12, "multiplication wrong")
check(abs((a**3).data - 8.0) < 1e-12, "power wrong")
check(abs((a-b).data - 5.0) < 1e-12, "subtraction wrong")
check(abs((a/b).data + 2/3) < 1e-9, "division wrong")
check(abs((3*a).data - 6.0) < 1e-12, "number * Value needs __rmul__")
check(abs((3+a).data - 5.0) < 1e-12, "number + Value needs __radd__")
check(abs(Value(0.0).tanh().data) < 1e-12, "tanh(0) = 0")
check(abs(Value(1.0).exp().data - math.e) < 1e-9, "exp wrong")
check(abs(Value(-2.0).relu().data) < 1e-12, "relu(-2) = 0")

# gradients, checked numerically
cases = [
    (lambda v: v[0]*v[1] + v[0], [2.0,-3.0], "product plus sum"),
    (lambda v: (v[0]*v[1]).tanh(), [0.7,1.3], "tanh of product"),
    (lambda v: v[0]*v[0] + v[0]*v[1] + v[1]**3, [1.5,2.0], "repeated use / accumulation"),
    (lambda v: (v[0].exp() / (v[1] + 4.0)), [0.5,2.0], "exp over sum"),
    (lambda v: (v[0]*2 + 1).relu() * v[1], [0.8,-1.5], "relu chain"),
]
for fn, xs, name in cases:
    vals = [Value(x) for x in xs]
    out = fn(vals); out.backward()
    for i in range(len(xs)):
        def g(t, i=i):
            p = list(xs); p[i] = t
            return fn([Value(z) for z in p]).data
        n_ = numgrad(g, xs[i])
        check(rel(vals[i].grad, n_) < 1e-4,
              f"{name}: input {i} analytic {vals[i].grad:.6f} vs numerical {n_:.6f}")

# accumulation, explicitly
z = Value(3.0); (z+z+z).backward()
check(abs(z.grad - 3.0) < 1e-9, f"a value used three times should accumulate to 3, got {z.grad}")

# a deep chain must not lose gradient
x = Value(0.5); y = x
for _ in range(20): y = (y*0.95 + 0.05).tanh()
y.backward()
def deep(t):
    v = Value(t)
    for _ in range(20): v = (v*0.95 + 0.05).tanh()
    return v.data
check(rel(x.grad, numgrad(deep, 0.5)) < 1e-3, "a 20-deep chain should still differentiate correctly")
""",
        [
            "Start with __init__ storing data, grad=0.0, _backward=lambda:None, _prev, _op.",
            "Every operation: compute out, define a closure that does `+=` into the inputs, attach it, return out.",
            "backward(): recursive topological build with a visited set, then seed grad=1.0 and walk reversed.",
            "Do not forget __radd__ and __rmul__, or `3*a` fails.",
            "Division reuses __pow__ with -1. Subtraction reuses __neg__.",
        ],
        """
import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data); self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children); self._op = _op

    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _b(): self.grad += out.grad; other.grad += out.grad
        out._backward = _b; return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _b():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _b; return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f'**{k}')
        def _b(): self.grad += k * (self.data ** (k-1)) * out.grad
        out._backward = _b; return out

    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-(o if isinstance(o,Value) else Value(o)))
    def __rsub__(self, o): return Value(o) + (-self)
    def __truediv__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return self * o**-1
    def __rtruediv__(self, o): return Value(o) * self**-1

    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,), 'tanh')
        def _b(): self.grad += (1 - t**2) * out.grad
        out._backward = _b; return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _b(): self.grad += out.data * out.grad
        out._backward = _b; return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _b(): self.grad += (1.0/self.data) * out.grad
        out._backward = _b; return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'relu')
        def _b(): self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _b; return out

    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self); self.grad = 1.0
        for n in reversed(topo): n._backward()

    __radd__ = __add__
    __rmul__ = __mul__
""",
    )
