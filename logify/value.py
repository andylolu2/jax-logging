import dataclasses
from typing import Any

import jax._src.tree_util as jtu
from jax._src.typing import Array

from logify.utils.accumulators import accumulators


@jtu.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Logs:
    _values: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)

    def __repr__(self):
        return f"Logs({repr(self._values)})"

    def log(self, name: str, value: Array, reduction: str):
        if name not in self._values:
            self._values[name] = {}
        if reduction not in self._values[name]:
            self._values[name][reduction] = accumulators[reduction].base

        self._values[name][reduction] = accumulators[reduction].update(
            value, self._values[name][reduction]
        )

    def asdict(self):
        result = {}
        for name, reductions in self._values.items():
            result[name] = {}
            for reduction, value in reductions.items():
                result[name][reduction] = accumulators[reduction].reduce(value)
        return result

    def tree_flatten(self):
        return (self._values,), ()

    @classmethod
    def tree_unflatten(cls, metadata, data):
        return cls(data[0])
