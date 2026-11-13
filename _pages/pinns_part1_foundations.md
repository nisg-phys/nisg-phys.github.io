---
layout: page
title: "PINNs Part 1: Why Physics Belongs in the Loss"
permalink: /learning/pinns-part1-foundations/
---

Notes from Part 1 of a series on Physics-Informed Neural Networks (PINNs) in JAX. A PINN is a neural network $$u_\theta(x)$$ trained so that a differential equation holds. This part builds the pieces from scratch (a small network, the Adam optimiser, derivatives of the network with respect to its input) and ends with the question that motivates the whole series: if a network can fit the data perfectly, why do we need physics at all?

The first half answers that. The second half covers two JAX ideas that kept confusing me, forward versus reverse mode and `jit`, because a PINN leans on both.

## 1. The problem: a network that fits the data and still fails

The target is a function we know exactly:

$$
y(x) = \sin x + 0.1\,x^2, \qquad x \in [-5, 5]
$$

It is sampled at 200 noise-free points. The first 160 (up to $$x \approx 3$$) are for training and the last 40 are for testing. Since $$x$$ is sorted, the split is spatial. The test set asks the network to **extrapolate**, not to interpolate between points it has seen.

A small `tanh` network trained with Adam gives:

|                        | Error (MSE)              |
| ---------------------- | ------------------------ |
| Train region (`x < 3`) | about `1e-5` (excellent) |
| Test region (`x > 3`)  | about `1` (wrong)        |

That is a gap of about five orders of magnitude. The fit is excellent where there are data and wrong where there are none.

### Why it fails

The loss only looks at the training points. Beyond `x = 3` nothing constrains the network, so it continues in whatever direction its `tanh` units happen to take it. This is not overfitting in the usual sense. The network was never asked to be right there.

More neurons, more steps and a better optimiser do not fix it. All of them fit the same 160 points better, and none adds information about `x > 3`.

## 2. "Isn't extrapolation what regression is for?"

The data came from a known equation, so it is tempting to expect any model to pick that up. It does not work that way. I trained several models on the same data (train on `x < 3`, test on `x > 3`):

| Model                          | What it assumes       | Train MSE | Test MSE |
| ------------------------------ | --------------------- | --------- | -------- |
| Straight line `[1, x]`         | the answer is a line  | 1.2       | 1.0      |
| Right form `[1, x, x², sin x]` | the exact ingredients | ~1e-30    | ~1e-29   |
| Degree-9 polynomial            | any smooth curve      | 5e-8      | 0.29     |
| Neural network                 | nothing in particular | ~2e-5     | ~1       |

- **The line fails everywhere.** The true function is not a line, so even the training fit is poor.
- **The right form extrapolates perfectly**, because it was handed the true ingredients. Regression does not extrapolate well in general. It extrapolates well when the form it assumes is correct.
- **The polynomial and the network fit the training data and then drift off outside it.** They are flexible and know nothing about the true shape. 160 points are consistent with infinitely many functions. These agree on the data and disagree everywhere else, and the model picks one more or less arbitrarily.

The functional dependence existed in how the data were generated. It was never part of what the model was trained on.

### The trade-off

| Model                                   | Knows the form? | Flexible? | Outside the data                     |
| --------------------------------------- | --------------- | --------- | ------------------------------------ |
| Regression with the right form          | yes             | no        | great, if you guessed the form right |
| Regression with the wrong form          | no              | no        | wrong everywhere                     |
| Neural network / high-degree polynomial | no              | yes       | fits the data, unreliable outside it |

You can have the form without the flexibility, or the flexibility without the form. A PINN aims to get both.

## 3. Giving the model the missing knowledge

There are three ways to supply what the data do not:

1. **Choose the form yourself.** This works only if you already know the ingredients, which is often not possible.
2. **Collect more data** where you need predictions. This is often expensive or impossible.
3. **Give the model the equation.** This is the PINN.

A PINN adds the equation to the loss:

```
loss = data error  +  (how badly the equation is violated)^2
```

The second term is the **residual**. It needs **no labels**: it can be evaluated at any $$x$$, including where there are no data, so it constrains the network exactly where the data run out.

Here the target satisfies $$y'' + y = 0.2 + 0.1x^2$$, so the residual of the network is

$$
r_\theta(x) = u_\theta''(x) + u_\theta(x) - \left(0.2 + 0.1\,x^2\right)
$$

and $$r_\theta = 0$$ everywhere exactly when $$u_\theta$$ solves the equation. Adding the mean of $$r_\theta^2$$ over points spread across $$[-5, 5]$$, with the data loss unchanged, took the test MSE from about `1` to the order of `1e-5` in my run.

A differential equation, together with a few conditions, pins down the solution. It removes the "infinitely many functions fit the data" problem, because only functions that satisfy the equation survive. The network stays flexible but is steered towards the ones that obey the physics.

### The same lesson applies to derivatives

Fitting $$u$$ does not mean $$u'$$ and $$u''$$ are right. The data loss controls values, not derivatives, so $$u''$$ can be badly wrong even inside the training region (about 66% relative error in my run). Differentiating amplifies small wiggles in $$u$$. The residual loss acts directly on $$u''$$, so it controls the derivatives that the data loss ignores.

**In one line:** data tells the network what happens at the points you measured, and physics tells it what must be true everywhere else.

## 4. Forward mode versus reverse mode

A PINN differentiates in two different ways, so it matters which autodiff mode does which job.

For a function $$f:\mathbb{R}^n \to \mathbb{R}^m$$, the Jacobian $$J$$ has one row per output and one column per input. Autodiff never builds it in one go. Each pass gives one slice:

|                       | Forward (`jacfwd`)                           | Reverse (`jacrev`, `grad`)               |
| --------------------- | -------------------------------------------- | ---------------------------------------- |
| Idea                  | nudge one **input**, push the change forward | pick one **output**, push blame backward |
| One pass gives        | one **column** of $$J$$                      | one **row** of $$J$$                     |
| Passes for full $$J$$ | $$n$$ (number of inputs)                     | $$m$$ (number of outputs)                |
| Use when              | few inputs, many outputs                     | many inputs, few outputs                 |
| Cost                  | cheap in memory                              | must store forward intermediates         |

### A worked example

Take $$f(z_0, z_1) = (z_0^2,\; z_0 z_1,\; \sin z_1)$$ at $$z = (2, 3)$$. Its Jacobian is

```
J = [[ 4,  0   ],
     [ 3,  2   ],
     [ 0, -0.99]]
```

- **Forward**, with the direction $$v = (1, 0)$$ ("wiggle $$z_0$$ by 1"): carry that wiggle through every operation and read off $$(4, 3, 0)$$. That is the first **column**. A second pass with $$v = (0, 1)$$ gives the second column.
- **Reverse**, with the weighting $$w = (1, 0, 0)$$ ("I only care about output 0"): run the function once and keep the intermediates, then send sensitivity backward to get $$(4, 0)$$. That is the first **row**. Rows two and three need two more passes.

`jax.jacfwd` and `jax.jacrev` return the same matrix, and the choice is purely about cost. (Entries that print as `-0.` and `0.` differ only in the sign of zero.)

### Why `grad` is reverse mode

A loss is one number and a network has thousands of parameters, so the Jacobian is a single row. One backward pass gives the entire gradient. Forward mode would need one pass per parameter.

### How one backward pass gets every gradient

Take the loss $$L = \left(w_2 \tanh(w_1 x) - y\right)^2$$ with $$x = 1$$, $$y = 0.5$$, $$w_1 = 0.5$$, $$w_2 = 2$$. The forward pass stores every intermediate value. The backward pass then computes, for each node $$v$$, how much $$L$$ changes when $$v$$ changes, working from the loss back to the parameters:

| Node    | Forward value       | Backward: $$\partial L / \partial \text{node}$$ |
| ------- | ------------------- | ----------------------------------------------- |
| $$L$$   | $$r^2 = 0.180$$     | $$1$$                                           |
| $$r$$   | $$p - y = 0.424$$   | $$2r = 0.848$$                                  |
| $$p$$   | $$w_2 h = 0.924$$   | $$0.848$$                                       |
| $$w_2$$ |                     | $$0.848 \cdot h =$$ **0.392**                   |
| $$h$$   | $$\tanh a = 0.462$$ | $$0.848 \cdot w_2 = 1.697$$                     |
| $$a$$   | $$w_1 x = 0.500$$   | $$1.697 \cdot (1 - h^2) = 1.335$$               |
| $$w_1$$ |                     | $$1.335 \cdot x =$$ **1.335**                   |

Both gradients, $$(1.335,\ 0.392)$$, came out of one sweep. Two things make that work:

1. Everything downstream of a node is already summarised in one number by the time the sweep reaches it, so nothing is redone per parameter.
2. Because $$L$$ is a scalar, every one of those numbers is a single number, not a vector or a matrix.

A backward pass costs roughly two to three times one forward pass, however many parameters there are. That is why this is how every neural network is trained. The price is memory, since the stored forward values are needed on the way back.

### What this means for PINNs

- **Loss with respect to parameters:** one output, thousands of inputs. Reverse mode (`grad`).
- **Network output with respect to coordinates** $$(x, t, \dots)$$: one output, a handful of inputs, but you need second derivatives. The standard recipe is forward-over-reverse, `jacfwd(grad(u))`, which is what `jax.hessian` does.

With one input dimension the difference is negligible. For a PDE in several dimensions it is not.

## 5. `jit`: trace once, reuse afterwards

On the first call, JAX does not run your Python with real numbers. It runs the function once with **tracers**, placeholders that know only a shape and a dtype, and records the math operations into a graph. XLA compiles that graph, and the result is cached under the input **shapes and dtypes**. Later calls with the same shapes run the compiled code and skip your Python entirely.

Two consequences are easy to trip over.

### A new shape means a recompile

```python
f = jax.jit(lambda x: jnp.sin(x) * 2)

f(jnp.ones(200))   # trace + compile  (slow)
f(jnp.ones(200))   # cached           (fast)
f(jnp.ones(201))   # new shape -> trace + compile again
```

- Different **values** with the same shape do not recompile.
- A different **shape or dtype** does.
- In practice, keep batch shapes constant. The smaller last mini-batch is a common culprit.

### Python side effects run only while tracing

```python
@jax.jit
def g(x):
    print("inside g:", x)    # runs ONCE, during tracing
    return x * 2
```

Called five times with the same shape, this prints **once**, and it prints `Traced<ShapedArray(float32[3])>` instead of numbers, because a tracer has no values. The compiled code contains only the math, so the later calls never execute the `print`.

The same goes for appending to a list, incrementing a counter or writing a file: it happens once, then silently stops. This is why JAX functions must be **pure**, meaning the output depends only on the inputs. To print real values on every call, use `jax.debug.print("x = {}", x)`.

## Summary

1. **A network fits the points it sees and nothing else.** Fitting the data says nothing about where there are no data.
2. **Regression extrapolates only when its assumed form is right.** A flexible model with no form, such as a neural network or a high-degree polynomial, fits the data and then drifts.
3. **A PINN supplies the missing rule.** The equation's residual needs no labels, so it constrains the network everywhere, and it controls the derivatives the data loss ignores.
4. **Reverse mode for the loss, forward mode for coordinate derivatives.** One backward pass gives every parameter gradient.
5. **`jit` traces once.** A new shape recompiles, and Python side effects run only during tracing.
