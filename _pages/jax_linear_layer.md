---
layout: page
title: "A Linear Layer in JAX: Thinking in Shapes"
permalink: /learning/jax-linear-layer/
---

Step 1 of building a Transformer from scratch in JAX. Every weight matrix in a Transformer (the Q, K, V and O projections, the two feed-forward matrices, the final projection onto the vocabulary) is the same building block:

$$
y = x W + b
$$

So it is worth writing once, carefully. The code is short. What takes real understanding is the shapes: which axis is which, what `@` actually multiplies, and why the same four lines work for one vector, a batch of vectors, and a batch of sequences.

## 1. The code

```python
def init_linear(key, in_dim, out_dim):
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    w = random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)
    b = jnp.zeros((out_dim,))
    return {"w": w, "b": b}

def linear_forward(params, x):
    return x @ params["w"] + params["b"]
```

This fixes a convention used for the rest of the project:

- A layer's parameters are a dict, e.g. `{"w": ..., "b": ...}`. That dict is a pytree, so `jax.grad` and `jax.jit` handle it directly.
- `init_*(key, ...)` builds the parameters.
- `*_forward(params, x)` is a pure function. It reads no global state and never modifies `params`.

Flax, Haiku and Equinox all follow this pattern internally. Here it is written out by hand.

## 2. Shapes: how JAX sees an array

A JAX array has a **shape**, which is a tuple of axis lengths. The number of axes is its **rank**.

| Array                 | Shape       | Rank | What it means                     |
| --------------------- | ----------- | ---- | --------------------------------- |
| one input vector      | `(4,)`      | 1    | 4 features                        |
| a batch of inputs     | `(5, 4)`    | 2    | 5 examples, 4 features each       |
| a batch of sequences  | `(2, 7, 4)` | 3    | 2 sequences, 7 tokens, 4 features |
| the weight matrix `w` | `(4, 3)`    | 2    | 4 inputs in, 3 outputs out        |
| the bias `b`          | `(3,)`      | 1    | one number per output             |

Two things are easy to get wrong here.

**`(4,)` is not a row or a column.** It is just a list of 4 numbers with one axis. `(1, 4)` is a row matrix and `(4, 1)` is a column matrix, and they are three different objects. A linear layer wants `(4,)` or `(batch, 4)`, never `(4, 1)`.

**The feature axis is always last.** Everything before it (batch, sequence position, attention head) is bookkeeping about _how many_ vectors you have. The layer only ever acts on the last axis.

## 3. Why `x @ W` and not `W x`

In a physics or maths textbook you would write $$y = Wx$$ with $$x$$ a column vector, so $$W$$ has shape `(out_dim, in_dim)`.

JAX (like NumPy and PyTorch) puts the vector first: `x @ W`, with `W` shaped `(in_dim, out_dim)`. The reason is batching. If each example is a row, a batch is just rows stacked on top of each other, and one matrix product handles all of them at once:

$$
\underbrace{X}_{(5,\,4)} \; \underbrace{W}_{(4,\,3)} = \underbrace{Y}_{(5,\,3)}
$$

The rule to remember: **the inner dimensions must match and are summed over. The outer dimensions survive.**

```
(5, 4) @ (4, 3)  ->  (5, 3)
    ^     ^
    must match, and disappear
```

## 4. A worked example with numbers

Take `in_dim = 2`, `out_dim = 3`:

$$
x = \begin{pmatrix} 1 & 2 \end{pmatrix}, \qquad
W = \begin{pmatrix} 1 & 0 & 2 \\ 3 & 1 & -1 \end{pmatrix}, \qquad
b = \begin{pmatrix} 0.5 & 0 & 1 \end{pmatrix}
$$

Each output is the dot product of $$x$$ with one **column** of $$W$$:

$$
\begin{aligned}
y_1 &= 1 \cdot 1 + 2 \cdot 3 + 0.5 = 7.5 \\
y_2 &= 1 \cdot 0 + 2 \cdot 1 + 0 = 2 \\
y_3 &= 1 \cdot 2 + 2 \cdot (-1) + 1 = 1
\end{aligned}
$$

So `x` of shape `(2,)` becomes `y = [7.5, 2, 1]` of shape `(3,)`. Column $$j$$ of $$W$$ holds the weights for output $$j$$; row $$i$$ holds how much input feature $$i$$ contributes to every output.

Now add a second example and pass a batch:

$$
X = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}
\quad\Longrightarrow\quad
XW + b = \begin{pmatrix} 7.5 & 2 & 1 \\ 3.5 & 1 & 0 \end{pmatrix}
$$

Row 1 is exactly the answer from before. Row 2 is the second example, computed independently. **Rows never mix.** That is what makes the batch axis safe to add.

In index notation this is

$$
Y_{nj} = \sum_i X_{ni} \, W_{ij} + b_j
$$

where $$n$$ runs over examples, $$i$$ over input features and $$j$$ over outputs. The index $$i$$ is summed away, and $$n$$ and $$j$$ survive.

## 5. What `@` does with more axes

`jnp.matmul` (which is what `@` calls) follows three rules:

1. **1D @ 2D**: the vector is treated as a single row, and the result is 1D. `(4,) @ (4, 3) -> (3,)`.
2. **2D @ 2D**: an ordinary matrix product. `(5, 4) @ (4, 3) -> (5, 3)`.
3. **Higher rank @ 2D**: the **last** axis of `x` is contracted with the **first** axis of `W`. Every other axis of `x` is treated as a batch axis and carried through untouched. `(2, 7, 4) @ (4, 3) -> (2, 7, 3)`.

The script checks all three with the same parameters:

```python
x_vec       = jnp.ones((4,))        # one example
x_batch     = jnp.ones((5, 4))      # 5 examples
x_seq_batch = jnp.ones((2, 7, 4))   # 2 sequences of 7 tokens

linear_forward(params, x_vec).shape        # (3,)
linear_forward(params, x_batch).shape      # (5, 3)
linear_forward(params, x_seq_batch).shape  # (2, 7, 3)
```

The index form makes rule 3 obvious. For a batch of sequences:

$$
Y_{s t j} = \sum_i X_{s t i} \, W_{i j} + b_j
$$

$$s$$ (which sequence) and $$t$$ (which token) are just along for the ride. The same weights are applied to every token of every sequence. This is exactly what a Transformer needs, and it is why this layer needs **no `vmap`**: `matmul` already batches over leading axes.

If you like `einsum`, the whole layer is one line and the shape logic is written out explicitly:

```python
y = jnp.einsum("...i,ij->...j", x, params["w"]) + params["b"]
```

`...` means "any number of leading axes, left alone", `i` is summed, and `j` is kept.

## 6. How the bias gets added: broadcasting

After the product, `y` has shape `(5, 3)` but `b` has shape `(3,)`. JAX makes them compatible by **broadcasting**: line the shapes up from the right, and wherever one array is missing an axis, repeat it along that axis.

```
x @ W :  (5, 3)
b     :     (3,)   ->  treated as (1, 3), then repeated 5 times
result:  (5, 3)
```

The same happens for `(2, 7, 3) + (3,)`: the bias is copied to all 14 token positions. No loop, and no copy is actually made in memory.

This is also why the bias is `(out_dim,)` and not `(1, out_dim)` or `(batch, out_dim)`: one bias per output, shared by every example.

## 7. When the shapes are wrong

Most bugs at this stage are shape bugs, and JAX catches them immediately.

- **Weights stored the other way round.** If `W` is `(3, 4)` (the textbook $$Wx$$ layout), then `(5, 4) @ (3, 4)` fails because the inner dimensions, 4 and 3, don't match.
- **A column vector instead of a flat vector.** `(4, 1) @ (4, 3)` also fails: the inner dimensions are 1 and 4. Flatten it with `x.reshape(-1)` or `x[:, 0]`.
- **A silent mistake.** If `in_dim == out_dim`, a transposed `W` has the right shape and nothing fails. The layer just computes the wrong thing. Assertions on shapes won't catch this; a small hand-computed example like section 4 will.

## 8. Initialization

The weights start as uniform random numbers in $$[-a, a]$$ with

$$
a = \sqrt{\frac{6}{\text{in\_dim} + \text{out\_dim}}}
$$

This is Glorot (or Xavier) initialization. For `in_dim = 4`, `out_dim = 3`, $$a = \sqrt{6/7} \approx 0.926$$.

The reason for this number: a uniform distribution on $$[-a, a]$$ has variance $$a^2 / 3$$, which here equals $$2 / (\text{in} + \text{out})$$. Each output is a sum of `in_dim` terms, so its variance grows with `in_dim`; the gradients flowing backwards grow with `out_dim`. Scaling by both keeps the signal roughly the same size going forwards and backwards, so activations neither blow up nor shrink to zero as layers stack. The Transformer paper doesn't specify an initialization, but this is the standard default. The bias starts at zero.

## 9. Gradients have the same shape as the parameters

The script checks that gradients actually reach both `w` and `b`, using a toy mean-squared-error loss:

```python
def toy_loss(params, x, target):
    pred = linear_forward(params, x)
    return jnp.mean((pred - target) ** 2)

grads = jax.grad(toy_loss)(params, x_batch, target)
```

`grads` is a dict with the **same structure and shapes** as `params`: `grads["w"]` is `(4, 3)` and `grads["b"]` is `(3,)`. That is what "`jax.grad` works on pytrees" means in practice.

The shapes also follow directly from the matrix picture. Write $$G = \partial L / \partial Y$$, which has the same shape as $$Y$$, `(5, 3)`. Then

$$
\frac{\partial L}{\partial W} = X^{\top} G
\qquad
\underbrace{(4,\,5)}_{X^\top} \; \underbrace{(5,\,3)}_{G} \to (4,\,3)
$$

$$
\frac{\partial L}{\partial b} = \sum_{n} G_{n\,\cdot}
\qquad (5,\,3) \to (3,)
$$

The batch axis is summed away in both, because every example used the same `W` and `b`. That sum over the batch is the backwards version of broadcasting in section 6.

The two checks the script runs on the gradients, and will run on every layer that follows:

- **Norm greater than zero**: if a gradient is exactly zero, that parameter is disconnected from the loss.
- **No NaNs**: NaNs mean something numerically broke.

## 10. `jit` works unchanged

```python
fast_forward = jax.jit(linear_forward)
jnp.allclose(fast_forward(params, x_batch), linear_forward(params, x_batch))  # True
```

Because `linear_forward` is pure, `jax.jit` can trace it once and compile it. One detail connects back to shapes: **`jit` compiles once per input shape.** Calling it with `(5, 4)`, then `(2, 7, 4)`, then `(4,)` triggers three separate compilations. In a real training loop you keep the batch shape fixed so the model compiles only once.

## Takeaways

- The feature axis is last; everything before it is batch bookkeeping.
- `x @ W` with `W` shaped `(in, out)`: inner dimensions match and are summed, outer dimensions survive.
- `@` contracts the last axis of `x` with the first axis of `W` and carries all leading axes through, so one layer handles vectors, batches and sequences with no `vmap`.
- Bias broadcasting in the forward pass becomes a sum over the batch in the backward pass.
- Gradients have exactly the shapes of the parameters, so the pytree convention carries straight through training.
