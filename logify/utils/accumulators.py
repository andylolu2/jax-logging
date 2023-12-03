from typing import Any, Generic, TypeVar

import jax._src.tree_util as jtu
import jax.numpy as jnp

T_acc = TypeVar("T_acc")
T_value = TypeVar("T_value")
T_result = TypeVar("T_result")
T = TypeVar("T")


class Accumulator(Generic[T_acc, T_value, T_result]):
    static_shape: bool

    @staticmethod
    def base(value: T_value) -> T_acc:
        raise NotImplementedError()

    @staticmethod
    def update(value: T_value, acc: T_acc) -> T_acc:
        raise NotImplementedError()

    @staticmethod
    def reduce(acc: T_acc) -> T_result:
        raise NotImplementedError()


class Replace(Accumulator[T, T, T]):
    static_shape = True

    @staticmethod
    def base(value: T) -> T:
        return jtu.tree_map(lambda x: jnp.zeros_like(x), value)

    @staticmethod
    def update(value: T, acc: T) -> T:
        return value

    @staticmethod
    def reduce(acc: T) -> T:
        return acc


class Append(Accumulator[list[T], T, list[T]]):
    static_shape = False

    @staticmethod
    def base(value: T) -> list[T]:
        return []

    @staticmethod
    def update(value: Any, acc: list[T]) -> list[T]:
        return acc + [value]

    @staticmethod
    def reduce(acc: list[T]) -> list[T]:
        return acc


class Mean(Accumulator[tuple[T, int], T, T]):
    static_shape = True

    @staticmethod
    def base(value: T) -> tuple[T, int]:
        return jtu.tree_map(lambda x: jnp.zeros_like(x), value), 0

    @staticmethod
    def update(value: Any, acc: tuple[T, int]) -> tuple[T, int]:
        old_value, count = acc
        new_value = jtu.tree_map(
            lambda x, y: x + (y - x) / (count + 1), old_value, value
        )
        return new_value, count + 1

    @staticmethod
    def reduce(acc: tuple[T | None, int]) -> T | None:
        return acc[0]


class Sum(Accumulator[T, T, T]):
    static_shape = True

    @staticmethod
    def base(value: T) -> T:
        return jtu.tree_map(lambda x: jnp.zeros_like(x), value)

    @staticmethod
    def update(value: T, acc: T) -> T:
        return jtu.tree_map(lambda x, y: x + y, acc, value)

    @staticmethod
    def reduce(acc: T) -> T:
        return acc


accumulators: dict[str, type[Accumulator]] = {
    "replace": Replace,
    "append": Append,
    "mean": Mean,
    "sum": Sum,
}
