from typing import Annotated, Any, Optional

import torch
from pydantic import Field, validate_call
from torchmetrics.functional.multimodal.clip_iqa import _PROMPTS as IQA_PROMPTS
from torchmetrics.multimodal import CLIPImageQualityAssessment
from torchvision.transforms import ToTensor

from ..utils.anybase import AnyBase
from ..utils.dtypes import ImgLike, SDict
from ..utils.file import load_image_to_pillow
from ..utils.misc import finalize_device
from ..utils.progress import progress_bar
from .base import VlmResult


class ClipIqa(AnyBase):
    """
    Reference: [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html)
    """

    @validate_call
    def __init__(
        self,
        prompts: tuple[str | tuple[str, str], ...] = ("quality",),
        device: Optional[str] = None,
    ) -> None:
        self._prompts = prompts
        self._device = finalize_device(device)
        self._metric = CLIPImageQualityAssessment(  # ResNet
            prompts=prompts,
            compute_on_cpu=self._device == "cpu",
        ).to(self._device)
        self._preprocess = ToTensor()

    @torch.inference_mode()
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self,
        images: list[ImgLike],
        *,
        w: Annotated[float, Field(gt=0.0)] = 100.0,
        progress: bool = False,
        **kwargs: Any,
    ) -> VlmResult:
        scores = torch.empty((len(self._prompts), len(images)))
        with progress_bar(disable=not progress) as pbar:
            # Initialize progress bar
            task_id = pbar.add_task(f"{self.cname_}", total=len(images))

            for i, img in enumerate(images):
                img_tensor = (
                    self._preprocess(load_image_to_pillow(img))
                    .unsqueeze(0)
                    .to(self.device)
                )
                r = self._metric(img_tensor)
                if isinstance(r, dict):
                    for p, value in enumerate(r.values()):
                        scores[p, i] = self._process_result(value, w)[0]
                elif isinstance(r, torch.Tensor):
                    assert len(self._prompts) == 1
                    scores[0, i] = self._process_result(r, w)[0]
                else:
                    raise RuntimeError(f"Unexpected result type: {type(r)}")

                # Update progress bar
                pbar.advance(task_id)
        return VlmResult(scores=scores)

    @property
    def default_prompts(self) -> SDict[tuple[str, str]]:
        return IQA_PROMPTS.copy()

    @property
    def device(self) -> str:
        return self._device

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _process_result(self, result: torch.Tensor, w: float) -> list[float]:
        result = result.detach().clamp(0.0, 1.0) * w
        if result.ndim == 0:
            return [result.item()]
        elif result.ndim == 1:
            return result.tolist()
        else:
            raise ValueError(f"Unexpected number of dims: {result.ndim}")
