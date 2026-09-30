import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Annotated, Callable, Literal, Optional

import torch
from huggingface_hub.constants import HF_HUB_CACHE
from pydantic import Field, validate_call

from ..utils import logger
from ..utils.anybase import AnyBase
from ..utils.dtypes import ImgLike, PathLike
from ..utils.file import load_image_to_pillow
from ..utils.llm import ZERO_SHOT_COT_REASONING as ZSCR
from ..utils.llm import CoStar, Llm, LlmOutput, PromptTemplate
from ..utils.maths import clamp
from ..utils.misc import finalize_device, format_error
from ..utils.progress import progress_bar
from .base import VlmResult
from .vqa_prompts import (
    VQA_DIRECT_CONTEXT,
    VQA_DIRECT_OBJECTIVE,
    VQA_MULTI_PERSPECTIVES_CONTEXT,
    VQA_MULTI_PERSPECTIVES_OBJECTIVE,
    VqaDirectOutput,
    VqaMultiPerspectivesOutput,
)

warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm")


def _concurrent_vqa_scores(
    get_score: Callable[..., float],
    images: list[ImgLike],
    texts: list[str],
    w: float,
    *,
    max_workers: int,
    label: str,
    progress: bool,
) -> torch.Tensor:
    """Score every ``(image, text)`` pair concurrently, preserving order.

    Each pair is submitted as its own future to a bounded thread pool (LLM
    calls are I/O-bound; the :class:`Llm` client is thread-safe). Results are
    written to the ``(i, j)`` slot tagged on the future, so the output tensor
    is independent of completion order. ``as_completed`` drives the progress
    bar as each task finishes rather than in submission order.

    Per-call LLM output (response panel + spinner) is shown only when a single
    worker is in flight; under real concurrency the interleaved output is noise.
    """
    scores = torch.empty((len(texts), len(images)))
    total = len(texts) * len(images)
    if total == 0:
        return scores

    workers = min(max_workers, total)
    with progress_bar(disable=not progress) as pbar:
        task_id = pbar.add_task(label, total=total)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(get_score, img, txt, w): (i, j)
                for i, txt in enumerate(texts)
                for j, img in enumerate(images)
            }
            for future in as_completed(future_to_idx):
                i, j = future_to_idx[future]
                scores[i, j] = future.result()
                pbar.advance(task_id)
    return scores


class VqaEvaluator(AnyBase):
    @validate_call
    def __init__(self, *, device: Optional[str] = None) -> None:
        self._device = finalize_device(device)

    def __call__(self) -> VlmResult:
        raise NotImplementedError()

    @property
    def device(self) -> str:
        return self._device


class BlipVqa(VqaEvaluator):
    _TEMPLATE = 'Does this image show "{}"? Please answer yes or no.'

    @validate_call
    def __init__(
        self, device: Optional[str] = None, cache_dir: Optional[PathLike] = None
    ) -> None:
        from transformers import BlipForQuestionAnswering, BlipProcessor

        super().__init__(device=device)
        self._processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-vqa-base", cache_dir=cache_dir
        )
        self._model = BlipForQuestionAnswering.from_pretrained(  # 224
            "Salesforce/blip-vqa-base", cache_dir=cache_dir
        ).to(self.device)
        self._yes_id: int = self._processor.tokenizer.encode(
            "yes", add_special_tokens=False
        )[0]
        self._no_id: int = self._processor.tokenizer.encode(
            "no", add_special_tokens=False
        )[0]

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self,
        images: list[ImgLike],
        texts: list[str],
        *,
        w: Annotated[float, Field(gt=0.0)] = 100.0,
        progress: bool = False,
    ) -> VlmResult:
        inputs = self._processor(
            [load_image_to_pillow(img) for img in images],
            [self._TEMPLATE.format(txt) for txt in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        scores = torch.empty((len(texts), len(images)))
        with progress_bar(disable=not progress) as pbar:
            # Initialize progress bar
            total = len(images) * len(texts)
            task_id = pbar.add_task(f"{self.cname_}", total=total)

            for txt_idx in range(len(texts)):
                for img_idx in range(len(images)):
                    scores[txt_idx, img_idx] = self._get_yes_prob(
                        inputs, txt_idx, img_idx, w
                    )

                    # Update progress bar
                    pbar.advance(task_id)
        return VlmResult(scores=scores)

    @torch.inference_mode()
    def _get_yes_prob(
        self, inputs, txt_idx: int, img_idx: int, w: float
    ) -> torch.Tensor:
        outputs = self._model.generate(
            input_ids=inputs.input_ids[txt_idx : txt_idx + 1],
            pixel_values=inputs.pixel_values[img_idx : img_idx + 1],
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        logits = outputs.scores[0].detach()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        prob_yes = probs[0, self._yes_id]
        prob_no = probs[0, self._no_id]
        total = prob_yes + prob_no
        if total == 0:
            return torch.tensor(0.0)
        return (prob_yes / total).clamp(0.0, 1.0).cpu() * w


class VqaDirect(VqaEvaluator):
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        llm: Llm,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        max_workers: Annotated[int, Field(ge=1)] = 8,
    ) -> None:
        super().__init__()
        self._llm = llm
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._max_workers = max_workers

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self,
        images: list[ImgLike],
        texts: list[str],
        *,
        w: Annotated[float, Field(gt=0.0)] = 100.0,
        progress: bool = False,
    ) -> VlmResult:
        scores = _concurrent_vqa_scores(
            self._get_score,
            images,
            texts,
            w,
            max_workers=self._max_workers,
            label=f"{self.cname_}",
            progress=progress,
        )
        return VlmResult(scores=scores)

    def _get_score(
        self, img: ImgLike, txt: str, w: float, verbose: bool = False
    ) -> float:
        prompt = CoStar.Json(
            context=VQA_DIRECT_CONTEXT,
            objective=PromptTemplate(VQA_DIRECT_OBJECTIVE, text=txt)(),
            response=VqaDirectOutput.to_str(reasoning=ZSCR),
        )
        llm_output: Optional[LlmOutput[VqaDirectOutput]] = None
        try:
            llm_output = self._llm(
                prompt,
                VqaDirectOutput,
                temperature=self._temperature,
                images=[img],
                reasoning_effort=self._reasoning_effort,
                verbose=verbose,
            )
        except Exception as e:
            logger.error(format_error(e))
        if llm_output is None or llm_output.response is None:
            return 0.0
        return clamp(llm_output.response.score, 0.0, 100.0) / 100.0 * w


class VqaMultiPerspectives(VqaEvaluator):
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        llm: Llm,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        max_workers: Annotated[int, Field(ge=1)] = 8,
    ) -> None:
        super().__init__()
        self._llm = llm
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._max_workers = max_workers

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self,
        images: list[ImgLike],
        texts: list[str],
        *,
        w: Annotated[float, Field(gt=0.0)] = 100.0,
        progress: bool = False,
    ) -> VlmResult:
        scores = _concurrent_vqa_scores(
            self._get_score,
            images,
            texts,
            w,
            max_workers=self._max_workers,
            label=f"{self.cname_}",
            progress=progress,
        )
        return VlmResult(scores=scores)

    def _get_score(
        self, img: ImgLike, txt: str, w: float, verbose: bool = False
    ) -> float:
        prompt = CoStar.Json(
            context=VQA_MULTI_PERSPECTIVES_CONTEXT,
            objective=PromptTemplate(VQA_MULTI_PERSPECTIVES_OBJECTIVE, text=txt)(),
            response=VqaMultiPerspectivesOutput.to_str(reasoning=ZSCR),
        )
        llm_output: Optional[LlmOutput[VqaMultiPerspectivesOutput]] = None
        try:
            llm_output = self._llm(
                prompt,
                VqaMultiPerspectivesOutput,
                temperature=self._temperature,
                images=[img],
                reasoning_effort=self._reasoning_effort,
                verbose=verbose,
            )
        except Exception as e:
            logger.error(format_error(e))
        if llm_output is None or llm_output.response is None:
            return 0.0
        score = mean(
            [
                llm_output.response.layout,
                llm_output.response.object,
                llm_output.response.texture,
                llm_output.response.physics,
                llm_output.response.functionality,
                llm_output.response.atmosphere,
            ]
        )
        return clamp(score, 0.0, 100.0) / 100.0 * w


class VqaScore(VqaEvaluator):
    @validate_call
    def __init__(
        self,
        model_name: Literal[
            "clip-flant5-xl",  # 336
            "clip-flant5-xxl",
            "llava-v1.5-7b",
            "llava-v1.5-13b",
            "llava-v1.6-13b",
            "instructblip-flant5-xl",
            "instructblip-flant5-xxl",
        ] = "clip-flant5-xl",
        device: Optional[str] = None,
        cache_dir: Optional[PathLike] = None,
    ) -> None:
        import t2v_metrics

        super().__init__(device=device)
        self._model = t2v_metrics.VQAScore(
            model=model_name,
            device=self.device,
            cache_dir=HF_HUB_CACHE if cache_dir is None else cache_dir,
        )
        if self.device != "cuda":
            self._model.model.model.to(device=self.device, dtype=torch.float32)

    @torch.inference_mode()
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self,
        images: list[PathLike],
        texts: list[str],
        *,
        w: Annotated[float, Field(gt=0.0)] = 100.0,
        progress: bool = False,
    ) -> VlmResult:
        scores = torch.empty((len(texts), len(images)))
        with progress_bar(disable=not progress) as pbar:
            # Initialize progress bar
            total = len(images) * len(texts)
            task_id = pbar.add_task(f"{self.cname_}", total=total)

            for i, txt in enumerate(texts):
                for j, img in enumerate(images):
                    scores[i, j] = (
                        self._model(images=[str(img)], texts=[txt])
                        .detach()
                        .clamp(0.0, 1.0)
                        .cpu()
                        * w
                    )[0]

                    # Update progress bar
                    pbar.advance(task_id)
        return VlmResult(scores=scores)
