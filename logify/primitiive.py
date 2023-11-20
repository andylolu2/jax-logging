import jax._src.tree_util as jtu
from jax._src import core

from logify.value import Logs

log_p = core.Primitive("log")
log_p.multiple_results = True  # zero results


@log_p.def_impl
def logs_impl(*log_values, log_tree):
    return []


@log_p.def_abstract_eval
def logs_abstract_eval(*log_values, log_tree):
    return []


def log_rule(prev_logs: Logs, *log_values, log_tree) -> tuple[list[core.Var], Logs]:
    new_logs = jtu.tree_unflatten(log_tree, log_values)
    for (name, reduction), v in new_logs.items():
        prev_logs.log(name, v, reduction)
    return [], prev_logs
