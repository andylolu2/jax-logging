from functools import partial
from typing import Any, Sequence

import jax._src.linear_util as lu
import jax._src.tree_util as jtu
from jax import lax
from jax._src import core, pjit, sharding_impls, source_info_util
from jax._src.interpreters import ad, batching, mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.util import safe_map, split_list

from logify.primitiive import log_p, log_rule
from logify.utils.accumulators import accumulators
from logify.value import Logs

map, unsafe_map = safe_map, map

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

    map(write_env, jaxpr.invars, in_args)
    map(write_env, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:
        invals = map(read_env, eqn.invars)
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack

        with source_info_util.user_context(
            eqn.source_info.traceback, name_stack=name_stack
        ):
            if eqn.primitive is log_p:
                outvals, logs = log_rule(logs, *invals, **eqn.params)
            elif eqn.primitive is lax.cond_p:
                outvals, logs = log_cond(logs, *invals, **eqn.params)
            elif eqn.primitive is lax.scan_p:
                outvals, logs = log_scan(logs, *invals, **eqn.params)
            elif eqn.primitive is lax.while_p:
                outvals, logs = log_while(logs, *invals, **eqn.params)
            elif eqn.primitive is pjit.pjit_p:
                outvals, logs = log_pjit(logs, *invals, **eqn.params)
            else:
                outvals = eqn.primitive.bind(*invals, **eqn.params)

        if eqn.primitive.multiple_results:
            map(write_env, eqn.outvars, outvals)
        else:
            write_env(eqn.outvars[0], outvals)

        core.clean_up_dead_vars(eqn, env, last_used)

    return map(read_env, jaxpr.outvars), logs


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
    new_jaxpr, out_vals, consts = pe.trace_to_jaxpr_dynamic(fun, args)
    new_jaxpr = core.ClosedJaxpr(new_jaxpr, consts)
    out_tree = out_tree_fn()

    return new_jaxpr, out_vals, out_tree


# Lowering (becomes No-op)
mlir.register_lowering(log_p, lambda ctx, *args, **kwargs: [])


# vmap-before-logify
def log_vmap(batched_args, batch_dims, *, log_tree):
    size = next(
        x.shape[dim]
        for x, dim in zip(batched_args, batch_dims)
        if dim is not batching.not_mapped
    )
    batched_args = (
        batching.bdim_at_front(a, d, size) for a, d in zip(batched_args, batch_dims)
    )
    log_p.bind(*batched_args, log_tree=log_tree)
    return [], ()


batching.primitive_batchers[log_p] = log_vmap


# jvp-before-logify
def log_jvp(primals, tangents, *, log_tree):
    log_p.bind(*primals, log_tree=log_tree)
    return [], []


ad.primitive_jvps[log_p] = log_jvp


# LAX control-flow primitives transformations


def _get_shaped_aval(val):
    return core.raise_to_shaped(core.get_aval(val))


# lax.cond
def log_cond(
    prev_logs: Logs,
    index,
    *args,
    branches: tuple[core.ClosedJaxpr, ...],
    linear: tuple[bool, ...],
):
    log_values, log_tree = jtu.tree_flatten(prev_logs)
    in_vals = [*log_values, *args]
    in_avals = map(_get_shaped_aval, in_vals)
    new_branches, _, out_trees = zip(
        *map(lambda jaxpr: jaxpr_to_logify_jaxpr(jaxpr, log_tree, *in_avals), branches)
    )
    new_linear = (False,) * len(log_values) + linear

    # assert all out_trees are the same
    if any(out_tree != out_trees[0] for out_tree in out_trees):
        raise ValueError("Logs must have the same shape for all conditional branches")

    out_vals = lax.cond_p.bind(index, *in_vals, branches=new_branches, linear=new_linear)

    out, logs = jtu.tree_unflatten(out_trees[0], out_vals)
    return out, logs


def _trace_logs(logs: Logs, jaxpr: core.ClosedJaxpr, *in_avals: core.AbstractValue):
    log_values, log_tree = jtu.tree_flatten(logs)
    in_avals = tuple(map(_get_shaped_aval, log_values)) + in_avals
    new_jaxpr, out_avals, out_tree = jaxpr_to_logify_jaxpr(jaxpr, log_tree, *in_avals)
    _, logs = jtu.tree_unflatten(out_tree, out_avals)
    return new_jaxpr, logs


def log_scan(
    prev_logs: Logs,
    *args,
    reverse: bool,
    length: int,
    jaxpr: core.ClosedJaxpr,
    num_consts: int,
    num_carry: int,
    linear: tuple[bool, ...],
    unroll: bool,
):
    consts, carry, xs = split_list(args, [num_consts, num_carry])
    xs_mapped = tuple(core.mapped_aval(length, 0, _get_shaped_aval(x)) for x in xs)

    # We need to initialize the accumulator such that it has static shapes.
    # Step 1: Identify logs keys used in the jaxpr.
    in_avals = tuple(map(_get_shaped_aval, [*consts, *carry])) + xs_mapped
    _, logs = _trace_logs(Logs(), jaxpr, *in_avals)

    # Step 2: Modify prev_logs to include new log keys.
    for name, reductions in logs.asdict().items():
        for reduction, value in reductions.items():
            if not accumulators[reduction].static_shape:
                raise ValueError(
                    f"Reduction '{reduction}' is not supported with lax.scan "
                    "since it uses dynamic shapes."
                )
            prev_logs.init(name, value, reduction)  # No-op if already initialized.

    # Step 3: Do actual pass with the new logs
    log_values, log_tree = jtu.tree_flatten(prev_logs)
    in_avals = tuple(map(_get_shaped_aval, [*log_values, *consts, *carry])) + xs_mapped
    new_jaxpr, _, out_tree = jaxpr_to_logify_jaxpr(jaxpr, log_tree, *in_avals)
    to_move = (
        [False] * len(log_values)
        + [True] * num_consts
        + [True] * num_carry
        + [False] * len(xs)
    )
    new_jaxpr = pe.move_binders_to_front(new_jaxpr, to_move)

    new_carry = (*carry, *log_values)
    new_args = (*consts, *new_carry, *xs)
    new_linear = (
        linear[: num_consts + num_carry]
        + (False,) * len(log_values)
        + linear[num_consts + num_carry :]
    )
    out_vals = lax.scan_p.bind(
        *new_args,
        reverse=reverse,
        length=length,
        jaxpr=new_jaxpr,
        num_consts=num_consts,
        num_carry=len(new_carry),
        linear=new_linear,
        unroll=unroll,
    )
    out, logs = jtu.tree_unflatten(out_tree, out_vals)

    return out, logs


def log_while(
    prev_logs: Logs,
    *args,
    cond_nconsts: int,
    cond_jaxpr: core.ClosedJaxpr,
    body_nconsts: int,
    body_jaxpr: core.ClosedJaxpr,
):
    c_consts, b_consts, carry = split_list(args, [cond_nconsts, body_nconsts])

    # We need to initialize the accumulator such that it has static shapes.
    # Step 1: Identify logs keys used in the jaxpr.
    # empty_log_values, empty_log_tree = jtu.tree_flatten(Logs())
    cond_in_avals = map(_get_shaped_aval, [*c_consts, *carry])
    _, logs = _trace_logs(Logs(), cond_jaxpr, *cond_in_avals)
    if len(logs._values) > 0:
        raise ValueError("Logs are not supported in the condition of lax.while_loop")

    body_in_avals = map(_get_shaped_aval, [*b_consts, *carry])
    _, logs = _trace_logs(logs, body_jaxpr, *body_in_avals)

    # Step 2: Modify prev_logs to include new log keys.
    for name, reductions in logs.asdict().items():
        for reduction, value in reductions.items():
            if not accumulators[reduction].static_shape:
                raise ValueError(
                    f"Reduction '{reduction}' is not supported with lax.while "
                    "since it uses dynamic shapes."
                )
            prev_logs.init(name, value, reduction)

    # Step 3: Do actual pass with the new logs
    log_values, log_tree = jtu.tree_flatten(prev_logs)
    cond_in_avals = map(_get_shaped_aval, [*log_values, *c_consts, *carry])
    new_cond_jaxpr, _, out_tree = jaxpr_to_logify_jaxpr(
        cond_jaxpr, log_tree, *cond_in_avals
    )
    to_move = [False] * len(log_values) + [True] * cond_nconsts + [True] * len(carry)
    new_cond_jaxpr = pe.move_binders_to_front(new_cond_jaxpr, to_move)
    new_cond_jaxpr = pe.prune_closed_jaxpr_outputs(
        new_cond_jaxpr, [True] * len(carry) + [False] * len(log_values)
    )

    body_in_avals = map(_get_shaped_aval, [*log_values, *b_consts, *carry])
    new_body_jaxpr, _, out_tree = jaxpr_to_logify_jaxpr(
        body_jaxpr, log_tree, *body_in_avals
    )
    to_move = [False] * len(log_values) + [True] * body_nconsts + [True] * len(carry)
    new_body_jaxpr = pe.move_binders_to_front(new_body_jaxpr, to_move)

    new_carry = (*carry, *log_values)
    new_args = (*c_consts, *b_consts, *new_carry)
    out_vals = lax.while_p.bind(
        *new_args,
        cond_nconsts=cond_nconsts,
        cond_jaxpr=new_cond_jaxpr,
        body_nconsts=body_nconsts,
        body_jaxpr=new_body_jaxpr,
    )
    out, logs = jtu.tree_unflatten(out_tree, out_vals)

    return out, logs


def log_pjit(
    prev_logs: Logs,
    *args,
    jaxpr: core.ClosedJaxpr,
    in_shardings,
    out_shardings,
    resource_env,
    donated_invars,
    name,
    inline,
    keep_unused,
):
    log_values, log_tree = jtu.tree_flatten(prev_logs)
    in_vals = [*log_values, *args]
    in_avals = tuple(map(_get_shaped_aval, in_vals))
    new_jaxpr, out_avals, out_tree = jaxpr_to_logify_jaxpr(jaxpr, log_tree, *in_avals)

    num_log_values = len(log_values)
    num_out_log_values = len(out_avals) - len(out_shardings)
    sharding = sharding_impls.UNSPECIFIED

    new_in_shardings = (*[sharding] * num_log_values, *in_shardings)
    new_out_shardings = (*[sharding] * num_out_log_values, *out_shardings)
    new_donated_invars = (*[False] * num_log_values, *donated_invars)

    out_vals = pjit.pjit_p.bind(
        *in_vals,
        jaxpr=new_jaxpr,
        in_shardings=new_in_shardings,
        out_shardings=new_out_shardings,
        resource_env=resource_env,
        donated_invars=new_donated_invars,
        name=name,
        inline=inline,
        keep_unused=keep_unused,
    )
    out, logs = jtu.tree_unflatten(out_tree, out_vals)

    return out, logs
