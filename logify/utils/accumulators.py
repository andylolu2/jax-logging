from typing import Any, Generic, TypeVar

import jax._src.tree_util as jtu

T_acc = TypeVar("T_acc")
T_value = TypeVar("T_value")
T = TypeVar("T")


class Accumulator(Generic[T_acc, T_value]):
    base: T_acc

    @staticmethod
    def update(value: Any, acc: T_acc) -> T_acc:
        raise NotImplementedError()

    @staticmethod
    def reduce(acc: T_acc) -> T_value:
        raise NotImplementedError()


class Replace(Accumulator[Any | None, Any]):
    base: Any | None = None

    @staticmethod
    def update(value: Any, acc: Any | None) -> Any | None:
        return value

    @staticmethod
    def reduce(acc: Any | None) -> Any | None:
        return acc


class Append(Accumulator[list[T], list[T]]):
    base: list[T] = []

    @staticmethod
    def update(value: Any, acc: list[T]) -> list[T]:
        return acc + [value]

    @staticmethod
    def reduce(acc: list[T]) -> list[T]:
        return acc


class Mean(Accumulator[tuple[T | None, int], T | None]):
    base: tuple[T | None, int] = (None, 0)

    @staticmethod
    def update(value: Any, acc: tuple[T | None, int]) -> tuple[T | None, int]:
        old_value, count = acc
        if old_value is None:
            new_value = value
        else:
            new_value = jtu.tree_map(
                lambda x, y: x + (y - x) / (count + 1), old_value, value
            )
        return new_value, count + 1

    @staticmethod
    def reduce(acc: tuple[T | None, int]) -> T | None:
        return acc[0]


class Sum(Accumulator[T | None, T | None]):
    base: T | None = None

    @staticmethod
    def update(value: Any, acc: T | None) -> T | None:
        if acc is None:
            return value
        else:
            return jtu.tree_map(lambda x, y: x + y, acc, value)

    @staticmethod
    def reduce(acc: T | None) -> T | None:
        return acc


accumulators: dict[str, type[Accumulator]] = {
    "replace": Replace,
    "append": Append,
    "mean": Mean,
    "sum": Sum,
}
