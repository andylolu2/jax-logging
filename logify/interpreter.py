from typing import Any, Sequence

from jax._src import core, source_info_util
from jax._src.interpreters import ad, batching, mlir
from jax._src.util import safe_map

from logify.primitiive import log_p, log_rule
from logify.value import Logs

# Become NO-OP when transformed by other tracers first.
# Logify the original function first to avoid this.
mlir.register_lowering(log_p, lambda ctx, *args, **kwargs: [])
batching.primitive_batchers[log_p] = lambda *args, **kwargs: ([], ())
ad.primitive_jvps[log_p] = lambda *args, **kwargs: ([], [])


def logify_jaxpr(
    jaxpr: core.Jaxpr, consts: Sequence[core.Value], logs: Logs, *args: core.Value
) -> tuple[Sequence[Any], Logs]:
    env: dict[core.Var, Any] = {}

    last_used = core.last_used(jaxpr)

    def read_env(var: core.Atom):
        if isinstance(var, core.Literal):
            return var.val
        return env[var]

    def write_env(var: core.Var, val: Any):
        env[var] = val

    safe_map(write_env, jaxpr.invars, args)
    safe_map(write_env, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:
        invals = safe_map(read_env, eqn.invars)
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack

        with source_info_util.user_context(
            eqn.source_info.traceback, name_stack=name_stack
        ):
            if eqn.primitive is log_p:
                outvals, logs = log_rule(logs, *invals, **eqn.params)
            else:
                outvals = eqn.primitive.bind(*invals, **eqn.params)

        if eqn.primitive.multiple_results:
            safe_map(write_env, eqn.outvars, outvals)
        else:
            write_env(eqn.outvars[0], outvals)

        core.clean_up_dead_vars(eqn, env, last_used)

    return safe_map(read_env, jaxpr.outvars), logs
