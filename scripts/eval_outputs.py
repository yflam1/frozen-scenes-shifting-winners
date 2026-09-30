import argparse
import ast
import inspect
import re
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field
from tqdm import tqdm

from vlmunr.metrics.base import VlmResult
from vlmunr.templates import DEFAULT_TEMPLATE as DT
from vlmunr.templates import TEMPLATES
from vlmunr.utils.dtypes import NonEmptyStr, SDict
from vlmunr.utils.file import load_json, save_json


class LlmConfig(BaseModel, validate_assignment=True):
    model: NonEmptyStr | None = None
    max_retries: Annotated[int, Field(ge=0)] = 10
    max_tokens: Annotated[int, Field(ge=1)] = 4096
    timeout: Annotated[float, Field(gt=0.0)] = 60.0
    base_url: str | None = None


class EvalOutput(BaseModel, validate_assignment=True):
    created_at: NonEmptyStr = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y%m%d-%H%M%S"), frozen=True
    )
    evaluator: NonEmptyStr
    kwargs: SDict[Any]
    base_dir: Path
    llm_config: LlmConfig
    results: Annotated[SDict[Any], Field(default_factory=dict)]

    def save(self, out_path: Path) -> None:
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True)
        save_json(self.model_dump(), out_path, indent=2)


def parse_kwargs(kwargs: list[str]) -> SDict[Any]:
    parsed: dict[str, Any] = {}
    for kwarg in kwargs:
        key, sep, value = kwarg.partition("=")
        if not sep:
            raise ValueError(f"Invalid kwarg {kwarg!r}: expected 'key=value'")
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid kwarg {kwarg!r}: empty key")
        value = value.strip()
        try:
            parsed[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed[key] = value
    return parsed


def evaluate(
    output: EvalOutput,
    evaluator: Callable[..., VlmResult],
    paraphrases: SDict[list[str]],
) -> Iterator[tuple[str, SDict[Any]]]:
    def apply_template(template: str, prompt: str) -> str:
        return template.replace("<QUERY>", prompt)

    config_jsons = sorted(output.base_dir.rglob("config.json"))
    for cj in tqdm(config_jsons, desc=output.evaluator):
        parent_key = str(cj.parent)
        if parent_key in output.results:
            # Already processed in a previous (interrupted) run; skip.
            continue
        data = load_json(cj)
        prompt = data.get("prompt", None) or data.get("query", None)
        if prompt is None:
            raise RuntimeError(f"Missing prompt in {cj}")
        result = {"prompt": prompt, "baseline": 0.0, "stages": {}}
        parent = cj.parent

        # Stages 1 and 2
        stages = [
            (
                "stage1a",  # Resolution
                "base/renderings/render_res-*_focal-50_pitch-0_yaw-0_env-city_bg-255-255-255.png",
            ),
            (
                "stage1b",  # Focal length
                "base/renderings/render_res-512_focal-*_pitch-0_yaw-0_env-city_bg-255-255-255.png",
            ),
            (
                "stage1c",  # Environment map (lighting)
                "base/renderings/render_res-512_focal-50_pitch-0_yaw-0_env-*_bg-255-255-255.png",
            ),
            (
                "stage1d",  # Background color
                "base/renderings/render_res-512_focal-50_pitch-0_yaw-0_env-city_bg-*-*-*.png",
            ),
            (
                "stage2a",  # Pitch
                "base/renderings/render_res-512_focal-50_pitch-*_yaw-0_env-city_bg-255-255-255.png",
            ),
            (
                "stage2b",  # Yaw
                "base/renderings/render_res-512_focal-50_pitch-45_yaw-*_env-city_bg-255-255-255.png",
            ),
        ]
        for stage, pattern in stages:
            imgs = sorted(parent.glob(pattern))
            assert len(imgs) > 0
            scores = evaluator(images=imgs, texts=[apply_template(DT, prompt)]).scores
            name_pattern = Path(pattern).name
            regex = re.compile(
                "(.+?)".join(re.escape(part) for part in name_pattern.split("*"))
            )
            result["stages"][stage] = {
                "-".join(regex.match(img.name).groups()): score.item()
                for img, score in zip(imgs, scores.squeeze(dim=0))
            }
            if stage == "stage1a":
                result["baseline"] = result["stages"][stage]["512"]

        if not output.evaluator.lower().startswith("clipiqa"):
            # Stage 3A (templates)
            imgs = sorted(
                parent.glob(
                    "base/renderings/render_res-512_focal-50_pitch-0_yaw-0_env-city_bg-255-255-255.png"
                )
            )
            assert len(imgs) == 1
            scores = evaluator(
                images=imgs,
                texts=[apply_template(template, prompt) for template in TEMPLATES],
            ).scores
            result["stages"]["stage3a"] = {
                template: score.item()
                for template, score in zip(TEMPLATES, scores.squeeze(dim=1))
            }

            # Stage 3B (paraphrases)
            pp = paraphrases.get(prompt, None)
            assert pp is not None
            texts = sorted(pp + [prompt])
            scores = evaluator(
                images=imgs, texts=[apply_template(DT, p) for p in texts]
            ).scores
            result["stages"]["stage3b"] = {
                text: score.item() for text, score in zip(texts, scores.squeeze(dim=1))
            }

        # Stage 4 (variants)
        result["stages"]["stage4"] = {}
        variants = ["half", "biggest-only", "scrambled", "worst-object"]
        for variant in variants:
            imgs = sorted(
                parent.glob(
                    f"variant_*_{variant}/renderings/render_res-512_focal-50_pitch-0_yaw-0_env-city_bg-255-255-255.png"
                )
            )
            assert len(imgs) == 1
            scores = evaluator(images=imgs, texts=[apply_template(DT, prompt)]).scores
            result["stages"]["stage4"][variant] = scores.squeeze(dim=0).item()

        yield (str(parent), result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", type=str, required=True)
    parser.add_argument("--kwarg", type=str, action="append", default=[])
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--llm-max-retries", type=int, default=10)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument("--llm-api-key", type=str, default=None)
    args = parser.parse_args()
    if not args.base_dir.exists() or not args.base_dir.is_dir():
        raise NotADirectoryError(f"{args.base_dir} is not a valid directory")
    if not args.prompts_json.exists() or not args.prompts_json.is_file():
        raise FileNotFoundError(f"{args.prompts_json} is not a valid file")
    kwargs = parse_kwargs(args.kwarg)
    output = EvalOutput(
        evaluator=args.evaluator,
        kwargs=kwargs,
        base_dir=args.base_dir,
        llm_config=LlmConfig(
            model=args.llm_model,
            max_retries=args.llm_max_retries,
            max_tokens=args.llm_max_tokens,
            timeout=args.llm_timeout,
            base_url=args.llm_base_url,
        ),
    )
    if args.out_path is None:
        args.out_path = (
            args.base_dir / f"eval-results_{args.evaluator}_{output.created_at}.json"
        )

    # Resume from a previous (interrupted) run: load any already-processed results
    # so they are skipped by evaluate().
    if args.out_path.exists() and args.out_path.is_file():
        prior = load_json(args.out_path)
        if prior.get("results"):
            output.results = prior["results"]
            print(f"Resuming: loaded {len(output.results)} runs from {args.out_path}")

    output.save(args.out_path)

    # Extract paraphrases
    paraphrases: SDict[list[str]] = {}
    prompts = load_json(args.prompts_json)
    for entries in prompts.values():
        for entry in entries:
            prompt = entry["prompt"]
            paraphrases[prompt] = [pp["paraphrase"] for pp in entry["paraphrases"]]

    # Load evaluator
    names = args.evaluator.split(".")
    names[0] = names[0].lower()
    if names[0] == "blipscore":
        from vlmunr.metrics.blipscore import BlipScore

        evaluator = BlipScore(**kwargs)
    elif names[0] == "clipiqa":
        from vlmunr.metrics.clipiqa import ClipIqa

        evaluator = ClipIqa(**kwargs)
    elif names[0] == "clipscore":
        assert len(names) == 2
        from vlmunr.metrics.clipscore import ClipScore

        evaluator = getattr(ClipScore, names[1])(**kwargs)
    elif names[0] == "vqascore":
        assert len(names) == 2
        from vlmunr.metrics import vqascore
        from vlmunr.utils.llm import Llm

        cls = getattr(vqascore, names[1])
        sig = inspect.signature(cls)
        if "llm" in sig.parameters:
            llm = Llm(
                args.llm_model,
                max_tokens=args.llm_max_tokens,
                timeout=args.llm_timeout,
                max_retries=args.llm_max_retries,
                api_key=args.llm_api_key,
                base_url=args.llm_base_url,
            )
            evaluator = cls(llm=llm, **kwargs)
        else:
            evaluator = cls(**kwargs)
    else:
        raise ValueError(f"Unknown evaluator: {args.evaluator}")

    for parent, result in evaluate(output, evaluator, paraphrases):
        output.results[parent] = result
        output.save(args.out_path)


if __name__ == "__main__":
    main()
