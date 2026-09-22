---
layout: page
title: "JAX Pytrees Primer"
permalink: /learning/jax-pytrees-primer/
---

## Why pytrees matter

You already know `jax.grad` / `jax.jit` / `jax.vmap` on plain arrays:

```python
def loss(w, x, y):
    pred = w @ x
    return jnp.mean((pred - y) ** 2)

grad(loss)(w, x, y)   # -> an array, same shape as w
```

A real model — say a Transformer — has dozens of weight matrices: per-layer Q/K/V/O projections, two feed-forward matrices, LayerNorm gains and biases, embeddings, and so on. You don't want to carry each one around as a separate function argument. Instead you bundle them into one nested Python structure of dicts, lists, and tuples, and JAX treats **any nested structure of containers full of arrays as a single differentiable object called a pytree.**

That's the whole idea. Everything below is just showing that it's true and covering the two or three functions you need to work with pytrees.

## 1. A pytree is just "arrays, possibly nested inside dict/list/tuple"

```python
params = {
    "layer1": {"w": jnp.ones((3, 2)), "b": jnp.zeros((2,))},
    "layer2": {"w": jnp.ones((2, 1)), "b": jnp.zeros((1,))},
}
```

**`jax.tree_util.tree_leaves`** pulls out every array, in a fixed order, discarding the dict/list structure:

```python
leaves = jax.tree_util.tree_leaves(params)
# [shape (3, 2), shape (2,), shape (2, 1), shape (1,)]
```

**`jax.tree_util.tree_map`** applies a function to every leaf and rebuilds the *same* nested structure around the results. This is the single most-used pytree function in practice:

```python
doubled = jax.tree_util.tree_map(lambda x: 2.0 * x, params)
doubled["layer1"]["w"][0, 0]  # == 2 * params["layer1"]["w"][0, 0]
```

`tree_map` also works elementwise across two pytrees **with the same structure** — this is exactly how you write "params minus learning-rate times grads":

```python
summed = jax.tree_util.tree_map(lambda a, b: a + b, params, doubled)
```

## 2. `jax.grad` works out of the box on pytree inputs

This is the "aha" moment: you never flatten `params` yourself to differentiate through it. `grad(loss)(params)` returns a pytree with the *exact same nested structure* as `params`, where each leaf is the derivative of the loss with respect to that leaf.

```python
def predict(params, x):
    h = jnp.tanh(x @ params["layer1"]["w"] + params["layer1"]["b"])
    y = h @ params["layer2"]["w"] + params["layer2"]["b"]
    return y

def loss_fn(params, x, y):
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)

grads = jax.grad(loss_fn)(params, x, y)

assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(params)
# grads["layer1"]["w"].shape == params["layer1"]["w"].shape, and so on for every leaf
```

## 3. A manual SGD step is one `tree_map`

This is what `optax.apply_updates` does internally, and it's how every "update the parameters" line in a JAX project looks, whether you write it by hand or hand it to optax:

```python
lr = 0.1
new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
```

One call, and every weight matrix and bias vector in the whole model gets its gradient step — no matter how deeply nested the model's parameter structure is.

## 4. `jit` and `vmap` also see straight through pytrees

Nothing special is needed — this is why "params as a nested dict" scales to a full Transformer without the code getting uglier.

```python
fast_loss = jax.jit(loss_fn)
fast_loss(params, x, y)   # works directly on the params pytree
```

`vmap` can even map over a *batch of parameter pytrees* (not something you'd need day to day, but proof it doesn't care about tree structure either). `in_axes=(0, None, None)` means "map over axis 0 of the first pytree argument, broadcast the other two":

```python
batched_params = jax.tree_util.tree_map(lambda p: jnp.stack([p, p]), params)
batched_loss = jax.vmap(loss_fn, in_axes=(0, None, None))(batched_params, x, y)
```

## Key takeaways

1. **Structure**: A pytree is any nested combination of dicts/lists/tuples with arrays at the leaves — that's what lets you bundle an entire model's parameters into one object.
2. **`tree_leaves`**: Flattens a pytree into its list of arrays, in a fixed order.
3. **`tree_map`**: Applies a function to every leaf (over one or more pytrees with matching structure) and rebuilds the same structure — this is the workhorse for parameter updates.
4. **`grad`**: Differentiates through a pytree input directly, returning a gradient pytree with the same structure — no manual flattening required.
5. **`jit` / `vmap`**: Both operate on pytrees transparently, so nothing about jit-compiling or batching changes as a model's parameter structure grows.

