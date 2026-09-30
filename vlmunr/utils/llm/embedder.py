from typing import Optional, Union, overload

import numpy as np
import torch
from pydantic import validate_call

from .base import OpenAiApi


class TextEmbedder(OpenAiApi):
    def __init__(
        self,
        model: str,
        *,
        max_retries: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        input_cost: Optional[float] = None,
    ) -> None:
        super().__init__(
            model=model,
            max_retries=max_retries,
            api_key=api_key,
            base_url=base_url,
            input_cost=input_cost,
        )

    @overload
    def __call__(
        self, text: str, response_type: type[torch.Tensor]
    ) -> torch.Tensor: ...

    @overload
    def __call__(
        self, text: str, response_type: type[np.ndarray]
    ) -> np.ndarray: ...

    @overload
    def __call__(
        self, text: str, response_type: type[list]
    ) -> list[float]: ...

    @overload
    def __call__(
        self, text: str, response_type: None = None
    ) -> list[float]: ...

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self,
        text: str,
        response_type: Optional[
            Union[type[torch.Tensor], type[np.ndarray], type[list]]
        ] = None,
    ) -> Union[torch.Tensor, np.ndarray, list]:
        response = self._client.embeddings.create(
            model=self._model, input=text
        )
        if self._input_cost_per_token is not None:
            self._cost += (
                response.usage.total_tokens * self._input_cost_per_token
            )
        assert len(response.data) > 0
        emb = response.data[0].embedding
        if response_type is None or response_type is list:
            return emb
        elif response_type is torch.Tensor:
            return torch.tensor(emb)
        elif response_type is np.ndarray:
            return np.array(emb)
        else:
            raise ValueError(f"Unsupported response_type: {response_type}")
