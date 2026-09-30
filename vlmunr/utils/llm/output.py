from datetime import datetime, timezone
from typing import Annotated, Any, Generic, Literal, Optional, TypeVar, Union
from uuid import uuid4

from openai.types.chat import ChatCompletionTokenLogprob
from pydantic import BaseModel, Field, HttpUrl

from ..dtypes import NonEmptyStr, SDict
from .msg import Messages
from .response import ResponseModel

# A response is either a parsed ``ResponseModel`` or the raw assistant string.
ResponseType = TypeVar("ResponseType", bound=Union[ResponseModel, str])


class LlmChatError(RuntimeError):
    """Raised for *semantic* failures of a completion (empty/None response,
    missing logprobs, …). These are retried locally; transport errors
    (rate-limit / 5xx / connection) are left to the OpenAI SDK's own retry
    machinery instead of being retried a second time here."""


class ChatParameters(BaseModel, validate_assignment=True, strict=True):
    model: NonEmptyStr
    max_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    timeout: Optional[Annotated[float, Field(gt=0.0)]] = None
    max_retries: Optional[Annotated[int, Field(ge=0)]] = None
    base_url: Optional[HttpUrl] = None
    temperature: Optional[Annotated[float, Field(ge=0.0, le=2.0)]] = None
    image_detail: Optional[Literal["auto", "low", "high"]] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[Annotated[int, Field(ge=0, le=5)]] = None
    extra_body: Optional[SDict[Any]] = None

    def to_api_params(self) -> SDict[Any]:
        # ``timeout``/``max_retries``/``base_url`` configure the client, not the
        # request; ``image_detail`` is applied when building the messages.
        return self.model_dump(
            exclude={"timeout", "max_retries", "base_url", "image_detail"},
            exclude_none=True,
        )


class LlmOutput(
    BaseModel, Generic[ResponseType], validate_assignment=True, strict=True
):
    id: NonEmptyStr = Field(default_factory=lambda: uuid4().hex, frozen=True)
    created_at: NonEmptyStr = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            r"%Y/%m/%d %H:%M:%S"
        ),
        frozen=True,
    )
    parameters: ChatParameters
    response: Optional[ResponseType] = None
    content: Optional[NonEmptyStr] = None
    detail: Optional[NonEmptyStr] = None
    duration: Optional[Annotated[float, Field(gt=0.0)]] = None
    finish_reason: Optional[NonEmptyStr] = None
    prompt_tokens: Optional[Annotated[int, Field(ge=0)]] = None
    completion_tokens: Optional[Annotated[int, Field(ge=0)]] = None
    cost: Optional[Annotated[float, Field(ge=0.0)]] = None
    logprobs: Optional[list[ChatCompletionTokenLogprob]] = None
    messages: Optional[Messages] = None
