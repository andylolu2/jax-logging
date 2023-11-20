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
print(json.dumps(logs, indent=2, default=str))
