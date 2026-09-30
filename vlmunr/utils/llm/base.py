import logging
import threading
from typing import Annotated, Optional

from openai import DEFAULT_MAX_RETRIES, OpenAI
from pydantic import Field, validate_call

from ..anybase import AnyBase
from ..dtypes import NonEmptyStr

logging.getLogger("httpx").setLevel(logging.WARNING)


class OpenAiApi(AnyBase):
    @validate_call
    def __init__(
        self,
        model: NonEmptyStr,
        *,
        timeout: Optional[Annotated[float, Field(gt=0.0)]] = None,
        max_retries: Optional[Annotated[int, Field(ge=0)]] = None,
        api_key: Optional[NonEmptyStr] = None,
        base_url: Optional[NonEmptyStr] = None,
        input_cost: Optional[float] = None,
        output_cost: Optional[float] = None,
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._max_retries = DEFAULT_MAX_RETRIES if max_retries is None else max_retries
        self._api_key = api_key
        self._base_url = base_url
        self._input_cost = input_cost
        self._output_cost = output_cost
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=self._max_retries,
        )
        self._input_cost_per_token = self._normalize_cost(input_cost)
        self._output_cost_per_token = self._normalize_cost(output_cost)
        self._cost = 0.0
        # The OpenAI/httpx client is itself thread-safe and shared across
        # conversations; only the running cost total needs guarding.
        self._cost_lock = threading.Lock()

    def __call__(self):
        raise NotImplementedError()

    @property
    def cost(self) -> float:
        with self._cost_lock:
            return self._cost

    def _add_cost(self, cost: float) -> float:
        """Atomically add to the running cost total and return the increment."""
        with self._cost_lock:
            self._cost += cost
        return cost

    def _normalize_cost(self, cost: Optional[float]) -> Optional[float]:
        if cost is None:
            return None
        return cost / 1_000_000
