from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Callable, Iterable, TypeVar

from pydantic import Field, validate_call

T = TypeVar("T")
R = TypeVar("R")


@validate_call(config=dict(arbitrary_types_allowed=True))
def concurrent_map(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    max_concurrency: Annotated[int, Field(ge=1)] = 8,
) -> list[R]:
    """Apply ``func`` to every item concurrently, preserving input order.

    LLM calls are I/O-bound, so a bounded thread pool is the right tool: it
    overlaps many in-flight requests while ``max_concurrency`` caps how many
    hit the provider at once (keeping you under server-side rate limits). The
    underlying OpenAI client and each :class:`Conversation` are thread-safe, so
    sharing them across the pool is safe.

    Results are returned in the same order as ``items``. The first exception
    raised by any task propagates once all submitted work has settled.

    Args:
        func: Callable invoked once per item.
        items: The inputs to map over.
        max_concurrency: Maximum number of concurrent in-flight calls.

    Returns:
        ``[func(item) for item in items]``, in order.
    """
    items = list(items)
    if len(items) == 0:
        return []
    workers = min(max_concurrency, len(items))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, items))
