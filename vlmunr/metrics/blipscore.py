import warnings
from os import makedirs
from os.path import dirname
from typing import Annotated, Literal, Optional

import torch
from lavis.models import load_model_and_preprocess
from pydantic import Field, validate_call

from ..utils.anybase import AnyBase
from ..utils.dtypes import ImgLike
from ..utils.file import load_image_to_pillow
from ..utils.misc import finalize_device
from ..utils.progress import progress_bar
from .base import VlmResult

warnings.filterwarnings("ignore", category=FutureWarning, module=r"lavis")


class BlipScore(AnyBase):
    @validate_call
    def __init__(
        self,
        model_type: Literal["coco", "pretrain"] = "pretrain",  # 224
        head: Literal["itc", "itm"] = "itm",
        device: Optional[str] = None,
    ) -> None:
        self._head = head
        self._device = finalize_device(device)
        self._model, self._vis_processors, self._text_processors = (
            load_model_and_preprocess(
                "blip2_image_text_matching",
                model_type,
                is_eval=True,
                device=self._device,
            )
        )

    @torch.inference_mode()
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self,
        images: list[ImgLike],
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

            # Preprocess images and texts
            imgs = [load_image_to_pillow(img) for img in images]
            processed_imgs = [
                self._vis_processors["eval"](img).unsqueeze(0).to(self.device)
                for img in imgs
            ]
            processed_txts = [self._text_processors["eval"](txt) for txt in texts]

            for i, txt in enumerate(processed_txts):
                for j, img in enumerate(processed_imgs):
                    output: torch.Tensor = self._model(
                        {"image": img, "text_input": txt},
                        match_head=self._head,
                    ).detach()
                    if self._head == "itc":
                        score = output
                    elif self._head == "itm":
                        score = output.softmax(dim=1)[:, 1]
                    scores[i, j] = score.clamp(0.0, 1.0).cpu()[0] * w

                    # Update progress bar
                    pbar.advance(task_id)
        return VlmResult(scores=scores)

    @property
    def device(self) -> str:
        return self._device


def _cli() -> None:
    import argparse
    import csv
    from glob import glob
    from itertools import product

    from ..utils.file import read_file
    from ..utils.misc import get_datetime, get_device

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", action="append", type=str, required=True)
    parser.add_argument("-pt", "--prompt_text", action="append", type=str, default=[])
    parser.add_argument("-pf", "--prompt_file", action="append", type=str, default=[])
    parser.add_argument("--head", type=str, default="itm")
    parser.add_argument(
        "-o",
        "--output_csv",
        type=str,
        default=f"blipscore_{get_datetime()}.csv",
    )
    args = parser.parse_args()
    if len(args.prompt_text) == 0 and len(args.prompt_file) == 0:
        raise ValueError(
            "At least one of `prompt_text` or `prompt_file` must be provided"
        )
    blip = BlipScore(head=args.head, device=get_device()[0])
    texts: list[str] = args.prompt_text
    for pf in args.prompt_file:
        paths = sorted(glob(pf))
        for path in paths:
            texts.extend(read_file(path, remove_spaces=True, remove_empty=True))
    output_dir = dirname(args.output_csv)
    if output_dir not in {"", "."}:
        makedirs(output_dir, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["head", "text", "image", "score"])
        for img in args.image:
            images = sorted(glob(img))
            result = blip(images, texts).scores
            for i, j in product(range(result.size(0)), range(result.size(1))):
                writer.writerow([args.head, texts[i], images[j], result[i, j].item()])


if __name__ == "__main__":
    _cli()
