---
layout: page
title: "Learning"
permalink: /learning/
nav: true
nav_order: 6
---

Occasional notes from things I am studying or trying to understand more clearly.

## JAX

**Core Concepts Summary** — A compact note covering key JAX ideas from the tutorial material.

- [Read the note](/learning/jax-tutorial-summary/)
- [View the repository](https://github.com/nisg-phys/Jax_tutorial)

**Adam from Scratch (Stage 1.5)** — Implementing the Adam optimizer directly from its mathematical definition in pure JAX, no Optax.

- [Read the note](/learning/adam-from-scratch-jax/)

**Pytrees Primer** — Why JAX treats any nested dict/list/tuple of arrays as a single differentiable object, and the handful of functions (`tree_leaves`, `tree_map`) you need to work with one.

- [Read the note](/learning/jax-pytrees-primer/)

**A Linear Layer: Thinking in Shapes** — The `y = xW + b` building block behind every Transformer weight matrix, with worked examples of how `@` handles vectors, batches and sequences, how the bias broadcasts, and why gradients come out the same shape as the parameters.

- [Read the note](/learning/jax-linear-layer/)

## Physics-Informed Neural Networks

**PINNs Part 1: Why Physics Belongs in the Loss** — Why a network that fits its data perfectly can still fail badly outside it, how linear regression, polynomials and neural networks compare at extrapolation, and how adding an equation's residual to the loss fixes it. Includes worked examples of forward versus reverse mode autodiff and of how `jit` traces and caches.

- [Read the note](/learning/pinns-part1-foundations/)
