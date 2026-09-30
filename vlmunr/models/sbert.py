from functools import cached_property
from typing import Optional

import torch
from pydantic import validate_call
from sentence_transformers import SentenceTransformer
from torch import Tensor

from ..utils.anybase import AnyBase
from ..utils.misc import finalize_device, loading


class SentenceBert(AnyBase):
    @validate_call
    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.__device = finalize_device(device)
        self.__model = SentenceTransformer(
            model_name, device=self.__device, cache_folder=cache_dir
        )

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(self, emb_1: Tensor, emb_2: Tensor) -> Tensor:
        if (d := emb_1.dim()) != 2:
            raise ValueError(f"Number of dims of `emb_1` must be 2, got {d}")
        if (d := emb_2.dim()) != 2:
            raise ValueError(f"Number of dims of `emb_2` must be 2, got {d}")

        # Dimensions: (N, D) @ (M, D) -> (N, M)
        return (emb_1 @ emb_2.T).clamp(-1.0, 1.0)

    @property
    def device(self) -> str:
        return self.__device

    @cached_property
    def dim(self) -> int:
        return self.encode("").size(dim=-1)

    @torch.inference_mode()
    @validate_call
    def encode(self, sentences: list[str], status: bool = False) -> Tensor:
        with loading(f"{self.cname_}...", disable=not status):
            return self.__model.encode(
                sentences,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device=self.device,
            ).detach()
