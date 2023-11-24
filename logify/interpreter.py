from functools import partial
from typing import Any, Sequence

import jax._src.linear_util as lu
import jax._src.tree_util as jtu
from jax import lax
from jax._src import core, source_info_util
from jax._src.interpreters import ad, batching, mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.util import safe_map, split_list

from logify.primitiive import log_p, log_rule
from logify.value import Logs

PyTreeDef = jtu.PyTreeDef

# Become NO-OP when transformed by other tracers first.
# Logify the original function first to avoid this.
mlir.register_lowering(log_p, lambda ctx, *args, **kwargs: [])
batching.primitive_batchers[log_p] = lambda *args, **kwargs: ([], ())
ad.primitive_jvps[log_p] = lambda *args, **kwargs: ([], [])


def logify_jaxpr(
    closed_jaxpr: core.ClosedJaxpr, logs: Logs, *args: core.Value
) -> tuple[Sequence[Any], Logs]:
    log_values, log_tree = jtu.tree_flatten(logs)
    return logify_jaxpr_flat(
        closed_jaxpr.jaxpr, closed_jaxpr.consts, log_tree, *log_values, *args
    )


def logify_jaxpr_flat(
    jaxpr: core.Jaxpr, consts: Sequence[Any], log_tree: PyTreeDef, *args: core.Value
) -> tuple[Sequence[Any], Logs]:
    env: dict[core.Var, Any] = {}
    log_values, in_args = split_list(args, [log_tree.num_leaves])
    logs = jtu.tree_unflatten(log_tree, log_values)

    last_used = core.last_used(jaxpr)

    def read_env(var: core.Atom):
        if isinstance(var, core.Literal):
            return var.val
        return env[var]

    def write_env(var: core.Var, val: Any):
        env[var] = val

    safe_map(write_env, jaxpr.invars, in_args)
    safe_map(write_env, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:
        invals = safe_map(read_env, eqn.invars)
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack

        with source_info_util.user_context(
            eqn.source_info.traceback, name_stack=name_stack
        ):
            if eqn.primitive is log_p:
                outvals, logs = log_rule(logs, *invals, **eqn.params)
            elif eqn.primitive is lax.cond_p:
                outvals, logs = log_cond(logs, *invals, **eqn.params)
            else:
                outvals = eqn.primitive.bind(*invals, **eqn.params)

        if eqn.primitive.multiple_results:
            safe_map(write_env, eqn.outvars, outvals)
        else:
            write_env(eqn.outvars[0], outvals)

        core.clean_up_dead_vars(eqn, env, last_used)

    return safe_map(read_env, jaxpr.outvars), logs


@lu.transformation_with_aux
def _flatten_and_get_out_tree(*invals):
    out, logs = yield invals, {}  # new_args, new_kwargs
    out_vals, out_tree = jtu.tree_flatten((out, logs))
    yield out_vals, out_tree  # new_outs, aux


def jaxpr_to_logify_jaxpr(closed_jaxpr: core.ClosedJaxpr, log_tree: PyTreeDef, *args):
    logify_jaxpr_partial = partial(
        logify_jaxpr_flat, closed_jaxpr.jaxpr, closed_jaxpr.consts, log_tree
    )

    fun = lu.wrap_init(logify_jaxpr_partial)
    fun, out_tree_fn = _flatten_and_get_out_tree(fun)
    new_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(fun, args)
    new_jaxpr = core.ClosedJaxpr(new_jaxpr, consts)
    out_tree = out_tree_fn()

    return new_jaxpr, out_tree


# LAX control-flow primitives transformations


# lax.cond
def log_cond(
    prev_logs: Logs,
    index,
    *args,
    branches: tuple[core.ClosedJaxpr, ...],
    linear: tuple[bool, ...],
):
    log_values, log_tree = jtu.tree_flatten(prev_logs)
    in_avals = tuple(
        map(lambda x: core.raise_to_shaped(core.get_aval(x)), [*log_values, *args])
    )
    new_branches, out_trees = zip(
        *map(lambda jaxpr: jaxpr_to_logify_jaxpr(jaxpr, log_tree, *in_avals), branches)
    )
    new_linear = (False,) * len(log_values) + linear

    # assert all out_trees are the same
    if any(out_tree != out_trees[0] for out_tree in out_trees):
        raise ValueError("Logs must have the same shape for all conditional branches")

    out_vals = lax.cond_p.bind(index, *args, branches=new_branches, linear=new_linear)

    out, logs = jtu.tree_unflatten(out_trees[0], out_vals)
    return out, logs
