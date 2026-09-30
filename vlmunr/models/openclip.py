from functools import cached_property
from typing import Iterable, Optional

import open_clip
import torch
from pydantic import validate_call
from torch import Tensor

from ..utils.anybase import AnyBase
from ..utils.dtypes import ImgLike
from ..utils.file import load_image_to_pillow
from ..utils.misc import finalize_device, loading


class OpenClip(AnyBase):
    @validate_call
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.__device = finalize_device(device)
        self.__model, _, self.__preprocess = (
            open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=self.__device,
                cache_dir=cache_dir,
            )
        )
        self.__tokenizer = open_clip.get_tokenizer(model_name)
        self.__model.eval()

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self, text_embedding: Tensor, img_embedding: Tensor
    ) -> Tensor:
        if (d := text_embedding.dim()) != 2:
            raise ValueError(
                f"Number of dims of `text_embedding` must be 2, got {d}"
            )
        if not 2 <= (d := img_embedding.dim()) <= 3:
            raise ValueError(
                f"Number of dims of `img_embedding` must be either 2 or 3, got {d}"
            )
        result = (
            torch.einsum("lkj, ij -> ilk", img_embedding, text_embedding)
            if img_embedding.dim() == 3
            else text_embedding @ img_embedding.T
        )
        return result.clamp(-1.0, 1.0)

    @property
    def device(self) -> str:
        return self.__device

    @cached_property
    def dim(self) -> int:
        return self.encode_texts("").size(dim=-1)

    @torch.inference_mode()
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def encode_images(
        self, images: ImgLike | Iterable[ImgLike], status: bool = False
    ) -> Tensor:
        with loading(f"{self.cname_}/images...", disable=not status):
            images = [images] if isinstance(images, ImgLike) else list(images)
            images = [load_image_to_pillow(img) for img in images]
            images = [self.__preprocess(img).unsqueeze(0) for img in images]
            stack = torch.vstack(images).to(self.__device)
            img_embed = self.__model.encode_image(stack)
            return img_embed / img_embed.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    @validate_call
    def encode_texts(
        self, texts: str | Iterable[str], status: bool = False
    ) -> Tensor:
        with loading(f"{self.cname_}/texts...", disable=not status):
            texts = [texts] if isinstance(texts, str) else list(texts)
            tokens = self.__tokenizer(texts).to(self.__device)
            text_embed = self.__model.encode_text(tokens)
            return text_embed / text_embed.norm(dim=-1, keepdim=True)
