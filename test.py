import json

import jax
import jax.numpy as jnp
from jax import lax

import logify


@jax.jit
@jax.vmap
@logify.logify
def f(a):
    # def sum_reduce(a, b):
    #     logify.log("a", a, reduction="mean")
    #     return a + b

    # a_sum = lax.reduce(a, 0, sum_reduce, [0])

    def yes():
        logify.log("a_yes", a)
        return a

    def no():
        logify.log("a_yes", a)
        return -a

    a_sum = lax.cond(a.sum() > 0, yes, no)

    return a_sum


a = jnp.arange(-3, 3)
# out = f(a)
out, logs = f(a)
logs = logs.asdict()

# print(jax.make_jaxpr(f)(a))
print(out)
print(json.dumps(logs, indent=2, default=str))
