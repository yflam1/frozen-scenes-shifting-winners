"""Torch-free VQA prompt templates and hosted response models.

These definitions are the request-format contract of
``vlmunr.metrics.vqascore``. They live in their own module so that
evidence-record guards can import the historical request format without
the heavy evaluator dependencies (torch, transformers, LAVIS) that
``vqascore`` itself pulls in. ``vqascore`` imports every name below, so
``vlmunr.metrics.vqascore.VqaDirectOutput`` and friends keep resolving.
"""

from typing import Annotated

from pydantic import Field

from ..utils.llm import JsonResponseModel

VQA_DIRECT_CONTEXT = """\
You are an expert in evaluating the alignment between images and texts."""

VQA_DIRECT_OBJECTIVE = """\
Evaluate to what extent the following text is correctly portrayed in the attached image with scale 0-100:
<text>
{text}
</text>
Provide a one-paragraph comprehensive and thorough reasoning on your decision."""

VQA_MULTI_PERSPECTIVES_CONTEXT = """\
You are an expert in evaluating the alignment between images and texts from multiple perspectives."""

VQA_MULTI_PERSPECTIVES_OBJECTIVE = """\
You are given the following text:
<text>
{text}
</text>
Evaluate the alignment between the text and the attached image from following perspectives, each with scale 0-100:
1. Layout: How well the spatial arrangement of objects in the image matches the description in the text.
2. Object: How accurately the objects mentioned in the text are represented in the image.
3. Texture: How well the visual texture details in the image matches the descriptions in the text.
4. Physics: How physically plausible objects in the image are positioned and oriented.
5. Functionality: How well the image demonstrates the intended use of objects as described in the text.
6. Atmosphere: How well the overall mood and tone of the image aligns with the text.
Provide a one-paragraph comprehensive and thorough reasoning on your decisions."""


class VqaDirectOutput(JsonResponseModel):
    reasoning: str
    score: Annotated[float, Field(ge=0.0, le=100.0)]


class VqaMultiPerspectivesOutput(JsonResponseModel):
    reasoning: str
    layout: Annotated[float, Field(ge=0.0, le=100.0)]
    object: Annotated[float, Field(ge=0.0, le=100.0)]
    texture: Annotated[float, Field(ge=0.0, le=100.0)]
    physics: Annotated[float, Field(ge=0.0, le=100.0)]
    functionality: Annotated[float, Field(ge=0.0, le=100.0)]
    atmosphere: Annotated[float, Field(ge=0.0, le=100.0)]
