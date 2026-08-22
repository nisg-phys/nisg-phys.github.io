---
layout: page
title: "Adam from Scratch in JAX"
permalink: /learning/adam-from-scratch-jax/
---

Stage 1.5 of a "from-scratch" JAX project: after building a small MLP and training it with plain gradient descent, the next step is to replace gradient descent with the Adam optimizer — implemented directly from its mathematical definition, with no Optax, Flax, or Equinox.

## 1. What Adam Is Solving

Given parameters \(\theta\) and loss \(L(\theta)\), autodiff gives the gradient \(g_t = \nabla_\theta L(\theta_t)\). Plain gradient descent applies a single learning rate \(\eta\) to every parameter:

$$
\theta_{t+1} = \theta_t - \eta\, g_t.
$$

That is inefficient when different parameters see very different gradient scales, gradients oscillate, or curvature varies a lot across directions. Adam keeps two moving averages per parameter:

- **First moment** \(m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t\) — behaves like momentum, tracking the recent gradient direction.
- **Second raw moment** \(v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2\) — tracks the recent scale of the gradient.

The update direction comes from \(m_t\); the step size is normalized by \(v_t\).

## 2. The Bias-Correction Problem

Starting from \(m_0 = v_0 = 0\), the first update is \(m_1 = (1-\beta_1) g_1\) — biased toward zero (and likewise for \(v_1\)). Adam corrects this with

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}.
$$

This is not optional polish — it's part of the standard algorithm, and it matters most in the first several steps.

## 3. The Adam Update Rule

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad
\hat v_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

| Hyperparameter | Typical value | Role |
|---|---:|---|
| \(\eta\) | `1e-3` | Learning rate |
| \(\beta_1\) | `0.9` | First-moment decay |
| \(\beta_2\) | `0.999` | Second-moment decay |
| \(\epsilon\) | `1e-8` | Numerical stability |

Adam produces a parameter-wise adaptive step size: a parameter with a historically large gradient scale gets a smaller normalized step.

## 4. Optimizer State

The key implementation difference from SGD is that **Adam is stateful**. SGD only needs current parameters and current gradients: \((\theta_t, g_t) \to \theta_{t+1}\). Adam carries state forward:

$$
(\theta_t, g_t, m_{t-1}, v_{t-1}, t) \;\to\; (\theta_{t+1}, m_t, v_t, t+1).
$$

For every parameter tensor, `m` and `v` must match its shape exactly, and optimizer state is kept separate from the model parameters themselves:

```text
AdamState
├── step
├── m
│   ├── W
│   └── b
└── v
    ├── W
    └── b
```

```python
def adam_init(params):
    m = [{"W": jnp.zeros_like(layer["W"]), "b": jnp.zeros_like(layer["b"])} for layer in params]
    v = [{"W": jnp.zeros_like(layer["W"]), "b": jnp.zeros_like(layer["b"])} for layer in params]
    return {"step": 0, "m": m, "v": v}
```

## 5. Building the Update Step by Step

**First moment** — exponential moving average of the gradient:

```python
def update_first_moment(m, grads, beta1):
    return [
        {"W": beta1 * m_l["W"] + (1 - beta1) * g_l["W"],
         "b": beta1 * m_l["b"] + (1 - beta1) * g_l["b"]}
        for m_l, g_l in zip(m, grads)
    ]
```

**Second moment** — exponential moving average of the *squared* gradient:

```python
def update_second_moment(v, grads, beta2):
    return [
        {"W": beta2 * v_l["W"] + (1 - beta2) * (g_l["W"] ** 2),
         "b": beta2 * v_l["b"] + (1 - beta2) * (g_l["b"] ** 2)}
        for v_l, g_l in zip(v, grads)
    ]
```

**Bias correction** — note `step` starts at 1 for the first update, not 0:

```python
def bias_correct(m, v, step, beta1, beta2):
    m_hat = [{"W": m_l["W"] / (1 - beta1 ** step), "b": m_l["b"] / (1 - beta1 ** step)} for m_l in m]
    v_hat = [{"W": v_l["W"] / (1 - beta2 ** step), "b": v_l["b"] / (1 - beta2 ** step)} for v_l in v]
    return m_hat, v_hat
```

**Parameter update** — the division by \(\sqrt{\hat v} + \epsilon\) is what makes the step adaptive:

```python
def apply_adam_update(params, m_hat, v_hat, learning_rate, eps):
    return [
        {"W": layer["W"] - learning_rate * (m_l["W"] / (jnp.sqrt(v_l["W"]) + eps)),
         "b": layer["b"] - learning_rate * (m_l["b"] / (jnp.sqrt(v_l["b"]) + eps))}
        for layer, m_l, v_l in zip(params, m_hat, v_hat)
    ]
```

**Putting it together** — the optimizer returns both new parameters and new state:

```python
def adam_update(params, grads, state, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    step = state["step"] + 1
    m = update_first_moment(state["m"], grads, beta1)
    v = update_second_moment(state["v"], grads, beta2)
    m_hat, v_hat = bias_correct(m, v, step, beta1, beta2)
    new_params = apply_adam_update(params, m_hat, v_hat, learning_rate, eps)
    new_state = {"step": step, "m": m, "v": v}
    return new_params, new_state
```

A useful engineering habit: test the optimizer primitive on a single step before dropping it into a 1,000-step training loop.

## 6. Training Loop and Architecture Separation

The optimizer never touches the model:

```text
Model → loss_fn → JAX autodiff → gradients → Adam → new parameters
```

```python
def train_adam(params, x, y, epochs=1000, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    state = adam_init(params)
    loss_history = []
    for epoch in range(epochs):
        loss, grads = loss_and_grad(params, x, y)
        params, state = adam_update(params, grads, state, learning_rate, beta1, beta2, eps)
        loss_history.append(loss)
    return params, state, jnp.array(loss_history)
```

Because `adam_update` only ever sees parameters, gradients, and optimizer state — never architecture — it works unchanged if the MLP grows another hidden layer, or later, when the loss itself contains derivatives with respect to physical coordinates (PINNs).

Once the algorithm is understood, the single training step can be JIT-compiled without hiding the algorithm behind a library:

```python
@jax.jit
def adam_train_step(params, state, x, y, learning_rate, beta1, beta2, eps):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    params, state = adam_update(params, grads, state, learning_rate, beta1, beta2, eps)
    return params, state, loss
```

## 7. SGD vs. Adam, and Why Bias Correction Matters

On the toy regression problem \(y = x^2\) (a 1→32→1 MLP with tanh), training both optimizers from the same initial parameters for 1,000 epochs gives comparable final loss (SGD at `lr=1e-2`, Adam at `lr=1e-3`) but different convergence shape — Adam is not automatically "better," it's an adaptive first-order method with extra state and hyperparameters, and the comparison is problem-dependent.

The bias-correction factors show why the correction is needed early on and fades out later. With \(\beta_1 = 0.9\), \(\beta_2 = 0.999\):

| step | \(1-\beta_1^t\) | \(1-\beta_2^t\) |
|---:|---:|---:|
| 1 | 0.100000 | 0.001000 |
| 2 | 0.190000 | 0.001999 |
| 5 | 0.409510 | 0.004990 |
| 10 | 0.651322 | 0.009955 |
| 100 | 0.999973 | 0.095208 |

At step 1 without correction, \(m_1\) is only 10% of \(g_1\); dividing by \(1-\beta_1^1 = 0.1\) recovers \(\hat m_1 = g_1\) exactly, which is the right value at the very first step. The second-moment correction factor converges far more slowly (\(\beta_2\) is closer to 1), which is why \(\epsilon\) in the denominator also matters — without it, a small or zero \(\hat v_t\) could blow up the step.

## Key Takeaways

1. **Adam has memory.** Unlike SGD, it carries `(m, v, step)` forward as explicit optimizer state, kept separate from model parameters.
2. **Two moving averages, two roles.** \(m_t\) estimates direction (momentum-like); \(v_t\) estimates recent gradient scale, used to normalize the step per parameter.
3. **Bias correction is not optional.** Without it, early updates are systematically too small, especially for \(v_t\) given \(\beta_2 \approx 1\).
4. **\(\epsilon\) is a small detail with outsized importance** — it prevents division by zero/near-zero in the adaptive denominator.
5. **The optimizer is architecture-agnostic.** `adam_update` only depends on the parameter pytree shape, gradients, and state — never on what produced them — which is exactly the separation needed before moving on to more sophisticated losses (PINNs) and higher-level libraries (Optax, Equinox).
