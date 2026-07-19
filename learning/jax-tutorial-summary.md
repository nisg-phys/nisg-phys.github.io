---
permalink: /learning/jax-tutorial-summary/
---

Source repository: [nisg-phys/Jax_tutorial](https://github.com/nisg-phys/Jax_tutorial)

## 1. JAX Arrays

**Core data structure**: Everything in JAX revolves around `jax.Array`.

**Key properties**:
- `jnp.array()` creates JAX arrays.
- The resulting array type is `jax.Array`.
- Arrays are immutable by default, so you do not modify them in place.
- Use `.at[index].set(value)` for indexed updates.

**Vectorized operations**:
- Supports element-wise operations such as `x + 2`, `x * 5`, and `jnp.sin(x)`.
- Broadcasting works in a NumPy-like way.

## 2. Pure Functions

**Definition**: A pure function:
- Always produces the same output for the same input.
- Has no side effects.
- Does not modify external state.
- Is self-contained.

**Example**:
```python
def cube(x):
    return x**3
```

## 3. Automatic Differentiation with `grad`

**Core function**: `jax.grad()` computes gradients.

**Usage**:
```python
from jax import grad

def f(x):
    return x**2

df = grad(f)
print(df(5.0))  # 10.0
```

**Advanced features**:
- Use `argnums` for multi-argument functions, for example `grad(f, argnums=1)`.
- Higher-order gradients are supported.

## 4. Function Composition with `value_and_grad`

**`value_and_grad`**: Computes both a function value and its gradient in one pass.

```python
from jax import value_and_grad

def f(x):
    return x**2

loss_and_grad = value_and_grad(f)
result, gradient = loss_and_grad(3.0)
```

**Batched gradient evaluation with `vmap`**:
```python
import jax
import jax.numpy as jnp

def loss(x):
    return x**2

dLdx = jax.grad(loss)
x_array = jnp.array([1.0, 2.0, 3.0])

# Vectorize gradient computation across a batch.
jax.vmap(dLdx)(x_array)
```

## 5. JIT Compilation

**Core function**: `jax.jit()` performs just-in-time compilation.

**Benefits**:
- Can significantly improve performance for repeated computations.
- Targets the available backend, such as CPU, GPU, or TPU.

**Best practice**: Use `jax.debug.print()` instead of regular `print()` inside JIT-compiled functions.

## 6. Vectorization with `vmap`

**Purpose**: Apply functions to batches of inputs in parallel.

**Example**:
```python
import jax
import jax.numpy as jnp

def square(x):
    return x**2

square_many = jax.vmap(square)
result = square_many(jnp.array([1.0, 2.0, 3.0, 4.0]))
```

**Advanced usage with `in_axes`**:
```python
import jax
import jax.numpy as jnp

def multiply(x, y):
    return x * y

x = jnp.array([1.0, 2.0, 3.0])

jax.vmap(multiply, in_axes=(0, None))(x, 10)  # Broadcast scalar input.
```

## 7. Jacobian Computation

**Functions**: `jax.jacrev()` for reverse-mode and `jax.jacfwd()` for forward-mode.

**Use cases**:
- Computing full Jacobian matrices.
- Understanding sensitivity of outputs to inputs.

**Example**:
```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.array([x[0] ** 2, x[0] + x[1]])

J = jax.jacrev(f)
jnp_input = jnp.array([1.0, 2.0])
jacobian_matrix = J(jnp_input)
```

## 8. Practical Example

**Simple neural network**:
```python
import jax
import jax.numpy as jnp

def network(x):
    W = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    return W @ x

J = jax.jacrev(network)
print(J(jnp.array([1.0, 2.0])))
```

## Key Takeaways

1. **Arrays**: Use immutable `jax.Array` values.
2. **Functions**: Keep functions pure so JAX transformations behave predictably.
3. **Differentiation**: Use `grad()` for scalar-output functions and `jacrev()` or `jacfwd()` for Jacobians.
4. **Performance**: Use `jit()` for compilation and `vmap()` for batch processing.
5. **Composition**: These transformations are designed to compose cleanly.
