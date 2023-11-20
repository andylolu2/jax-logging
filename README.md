# Imperative-like logging for JAX

`logify` (To do: Better name!) is a light-weight extension module to JAX that provides imperative-like metrics logging functionality. This allows you to log any intermediate values (e.g. metrics / activations) without having to explicitly pass them around as return values.

## Example

```python
# demo.py
import json

import jax
import jax.numpy as jnp

import logify


@jax.jit
@logify.logify
def f(a, b):
    # reduction="mean" takes the mean of the values logged.
    logify.log("b", b, reduction="mean")
    logify.log("b", b + 1, reduction="mean")
    logify.log("b", b + 1, reduction="mean")

    x = jnp.sum(a + b)

    # default is reduction="replace" which uses the last value logged.
    logify.log("y", x**2 - 1)
    logify.log("y", x**2)

    return x


a = jnp.arange(3)
b = jnp.zeros(3)
out, logs = f(a, b)
logs = logs.asdict()

print(out)
# 3.0
print(json.dumps(logs, indent=2, default=str))
# {
#   "b": {
#     "mean": "[0.6666667 0.6666667 0.6666667]"
#   },
#   "y": {
#     "replace": "9.0"
#   }
# }
```

## How it works

It really isn't that complicated! `logify` works by transforming the decorated function, making all logged values into return values. For example, the function:

```python
@logify.logify
def f(a, b):
    logify.log("b", b)
    return a + b
```

is effectively transformed into:

```python
def f(a, b):
    return a + b, {"b": {"replace": b}}
```

And so the typical JAX transformations (e.g. `jit`, `grad`, `vmap`) can be applied to the decorated function as usual. A more complicated example:

```python
@logify.logify
def f(a, b):
    logify.log("b", b, reduction="mean")
    logify.log("b", b + 1, reduction="mean")
    logify.log("b", b + 1, reduction="mean")

    x = jnp.sum(a + b)

    logify.log("y", x**2 -1)
    logify.log("y", x**2)

    return x
```

effectively becomes

```python
def mean_update(x, state):
    value, count = state
    if value is None:
        return x, 1
    else:
        return value + (x - value) / (count + 1), count + 1

def f(a, b):
    logs = {}
    logs["b"] = {"mean": (None, 0)}  # (value, count)
    logs["b"]["mean"] = mean_update(b, logs["b"]["mean"])
    logs["b"]["mean"] = mean_update(b + 1, logs["b"]["mean"])
    logs["b"]["mean"] = mean_update(b + 1, logs["b"]["mean"])

    x = jnp.sum(a + b)

    logs["y"] = {"replace": None}
    logs["y"]["replace"] = x**2 - 1
    logs["y"]["replace"] = x**2

    return x, logs
```

In practice, this transformation happens on the JAX interpreter level, not on the code level.

## Gotchas

- `logify.logify` should be applied before any other JAX transformations (e.g. `jit`, `grad`, `vmap`). Otherwise, the `logify.log` calls will become NO-OPs (i.e. the returned logs will be empty).
- Only implemented `replace`, `append`, `mean`, `sum` reductions. `replace` is the default.
- Currently, a NO-OP for control-flow `lax` operations like `lax.cond`, `lax.scan`, `lax.while_loop` etc. (Will be implemented soon!)
