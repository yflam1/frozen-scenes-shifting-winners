from typing import Iterable, TypeVar

from pydantic import validate_call

N = TypeVar("N", float, int)


@validate_call
def is_even(num: int) -> bool:
    return num % 2 == 0


@validate_call
def is_odd(num: int) -> bool:
    return num % 2 != 0


@validate_call
def rescale(num: N, a: N, b: N, c: N, d: N) -> N:
    """
    Map a number in [a, b] (a != b) to [c, d].
    """
    assert a < b and a <= num <= b, "num must be in [a, b] where a != b"
    assert c <= d, f"{c} > {d}. c must not be greater than d."
    return (num - a) * (d - c) / (b - a) + c


@validate_call
def clamp(num: N, a: N, b: N) -> N:
    """
    Restrict a number to a specific range.
    """
    assert a <= b, f"{a} > {b}. a must not be greater than b."
    return max(a, min(num, b))


@validate_call
def minmax(arr: Iterable[N]) -> tuple[N, N]:
    min = max = None
    for val in arr:
        if min is None or val < min:
            min = val
        if max is None or val > max:
            max = val
    return min, max


@validate_call
def next_multiple_of(num: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(
            f"`multiple` must be greater than 0, got {multiple} instead"
        )
    return ((num + multiple - 1) // multiple) * multiple
