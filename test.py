import json

import jax
import jax.numpy as jnp
from jax import lax

import logify


def test_cond():
    @jax.jit
    @jax.vmap
    @logify.logify
    def f(a):
        def yes():
            logify.log("a_yes", a)
            return a

        def no():
            logify.log("a_yes", a)
            return -a

        a_sum = lax.cond(a.sum() > 0, yes, no)

        return a_sum

    a = jnp.arange(-3, 3)
    out, logs = f(a)
    logs = logs.asdict()

    print(out)
    print(json.dumps(logs, indent=2, default=str))


def test_scan():
    @jax.jit
    @logify.logify
    def g():
        def sum_(a, b):  # compute a + b
            def f(carry, i):
                logify.log("i", i, "mean")
                return carry + 1, None

            return lax.scan(f, b, jnp.arange(a))[0]

        def prod(a, b):  # compute a * b
            def f(carry, j):
                logify.log("j", j, "mean")
                logify.log("j", j)
                return sum_(b, carry), None

            return lax.scan(f, 0, jnp.arange(a))[0]

        logify.log("j", -100, "mean")
        return prod(7, 3)

    out, logs = g()
    logs = logs.asdict()

    print(out)
    print(json.dumps(logs, indent=2, default=str))


def test_fori():
    @jax.jit
    @logify.logify
    def g():
        def sum_(a, b):  # compute a + b
            def f(i, carry):
                logify.log("i", i)
                return carry + 1

            return lax.fori_loop(0, a, f, b)

        def prod(a, b):  # compute a * b
            def f(j, carry):
                logify.log("prod_carry", carry)
                logify.log("j", j)
                return sum_(b, carry)

            return lax.fori_loop(0, a, f, 0)

        logify.log("j", -1)
        return prod(7, 2)

    out, logs = g()
    logs = logs.asdict()

    print(out)
    print(json.dumps(logs, indent=2, default=str))


def test_while():
    @logify.logify
    @jax.jit
    def g():
        def cond(carry):
            return carry < 10

        def body(carry):
            logify.log("carry", carry, "mean")
            logify.log("carry", carry)
            return carry + 1

        return lax.while_loop(cond, body, 0)

    out, logs = g()
    logs = logs.asdict()

    print(out)
    print(json.dumps(logs, indent=2, default=str))


if __name__ == "__main__":
    # test_cond()
    # test_scan()
    # test_fori()
    test_while()
