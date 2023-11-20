from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

from jax._src import linear_util as lu
from jax._src import tree_util as jtu
from jax._src.api_util import flatten_fun
from jax._src.interpreters import partial_eval as pe

from logify.interpreter import logify_jaxpr
from logify.primitiive import log_p
from logify.value import Logs

P = ParamSpec("P")
T = TypeVar("T")


def logify(f: Callable[P, T]) -> Callable[P, tuple[T, Logs]]:
    @wraps(f)
    def wrapped_f(*args: P.args, **kwargs: P.kwargs) -> tuple[T, Logs]:
        # close over all arguments so they're not turned into abstract values.
        in_tree = jtu.tree_structure(((), {}))
        closed_f = lambda: f(*args, **kwargs)

        # stage
        fun_, out_tree = flatten_fun(lu.wrap_init(closed_f), in_tree)
        debug = pe.debug_info(closed_f, in_tree, out_tree, False, "logify")
        jaxpr_, _, args = pe.trace_to_jaxpr_dynamic(fun_, (), debug)
        jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(jaxpr_))

        # logify
        out_flat, logs = logify_jaxpr(jaxpr.jaxpr, jaxpr.consts, Logs(), *args)
        return jtu.tree_unflatten(out_tree(), out_flat), logs

    return wrapped_f


def log(name: str, value: Any, reduction: str = "replace") -> None:
    log_values, log_tree = jtu.tree_flatten({(name, reduction): value})
    log_p.bind(*log_values, log_tree=log_tree)
