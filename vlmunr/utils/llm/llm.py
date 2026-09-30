import random
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import Annotated, Any, Iterator, Literal, Optional, Sequence, Union

from openai.types.chat import ChatCompletion
from pydantic import Field, validate_call
from rich.panel import Panel
from typing_extensions import Self

from .. import console
from ..dtypes import ImgLike, NonEmptyStr, SDict
from ..misc import format_error, loading
from .base import OpenAiApi
from .concurrency import concurrent_map
from .conversation import Conversation
from .msg import Messages
from .output import ChatParameters, LlmChatError, LlmOutput, ResponseType
from .response import ResponseModel, ResponseModelParsingError
from .template import PromptTemplate

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# rich's ``Live`` display (used by the loading spinner) allows only one active
# instance per process. Concurrent calls therefore share a single spinner: the
# first to grab this lock shows it, the rest run quietly.
_spinner_lock = threading.Lock()


@contextmanager
def _maybe_loading(desc: str, *, disable: bool) -> Iterator[None]:
    if disable or not _spinner_lock.acquire(blocking=False):
        yield
        return
    try:
        with loading(desc):
            yield
    finally:
        _spinner_lock.release()


class Llm(OpenAiApi):
    """A thread-safe client/factory over an OpenAI-compatible Chat Completions
    endpoint.

    The client itself is stateless with respect to dialogue: conversation state
    lives in :class:`Conversation` objects spawned via :meth:`conversation`.
    A single ``Llm`` (and its underlying OpenAI client) may be shared freely
    across threads — see :func:`concurrent_map` and :meth:`batch` for running
    many conversations at once.
    """

    @validate_call
    def __init__(
        self,
        model: str,
        *,
        max_tokens: Annotated[int, Field(ge=1)] = 4096,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        input_cost: Optional[float] = None,
        output_cost: Optional[float] = None,
        extra_body: Optional[SDict[Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            api_key=api_key,
            base_url=base_url,
            input_cost=input_cost,
            output_cost=output_cost,
        )
        self._max_tokens = max_tokens
        self._extra_body = deepcopy(extra_body) if extra_body is not None else None
        self._kwargs = kwargs

    # ---------------------------------------------------------------- factories

    @validate_call
    def conversation(
        self,
        *,
        sys_prompt: Optional[NonEmptyStr] = DEFAULT_SYSTEM_PROMPT,
        verbose: bool = True,
    ) -> Conversation:
        """Start a new, independent conversation with its own system prompt.

        Each conversation owns its own message history and lock, so distinct
        conversations are safe to drive from different threads concurrently.
        """
        return Conversation(self, sys_prompt=sys_prompt, verbose=verbose)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self,
        prompt: Union[NonEmptyStr, PromptTemplate],
        response_type: type[ResponseType] = str,
        *,
        sys_prompt: Optional[NonEmptyStr] = DEFAULT_SYSTEM_PROMPT,
        images: Optional[Sequence[ImgLike]] = None,
        image_detail: Literal["auto", "low", "high"] = "auto",
        temperature: Optional[Annotated[float, Field(ge=0.0, le=2.0)]] = None,
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[Annotated[int, Field(ge=0, le=5)]] = None,
        extra_body: Optional[SDict[Any]] = None,
        base_delay: Annotated[float, Field(ge=1.0)] = 1.5,
        max_parse_retries: Annotated[int, Field(ge=0)] = 2,
        verbose: bool = True,
    ) -> LlmOutput[ResponseType]:
        """Run a single, stateless completion (a fresh one-turn conversation)."""
        convo = self.conversation(sys_prompt=sys_prompt, verbose=verbose)
        return convo.send(
            prompt,
            response_type,
            images=images,
            image_detail=image_detail,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            extra_body=extra_body,
            base_delay=base_delay,
            max_parse_retries=max_parse_retries,
        )

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def batch(
        self,
        prompts: Sequence[Union[NonEmptyStr, PromptTemplate]],
        response_type: type[ResponseType] = str,
        *,
        max_concurrency: Annotated[int, Field(ge=1)] = 8,
        sys_prompt: Optional[NonEmptyStr] = DEFAULT_SYSTEM_PROMPT,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[LlmOutput[ResponseType]]:
        """Run many independent one-shot completions concurrently, in order.

        ``max_concurrency`` bounds how many requests are in flight at once so
        the provider does not rate-limit you. ``verbose`` defaults to ``False``
        here because interleaved panels from many threads are rarely useful.
        """

        def run(
            prompt: Union[NonEmptyStr, PromptTemplate],
        ) -> LlmOutput[ResponseType]:
            return self(
                prompt,
                response_type,
                sys_prompt=sys_prompt,
                verbose=verbose,
                **kwargs,
            )

        return concurrent_map(run, prompts, max_concurrency=max_concurrency)

    # ------------------------------------------------------------------ engine

    def build_params(
        self,
        *,
        temperature: Optional[float],
        image_detail: Literal["auto", "low", "high"],
        reasoning_effort: Optional[Literal["low", "medium", "high"]],
        logprobs: Optional[bool],
        top_logprobs: Optional[int],
        extra_body: Optional[SDict[Any]],
    ) -> ChatParameters:
        """Merge per-call overrides with the client's defaults into a
        validated :class:`ChatParameters`. Pure / thread-safe."""
        rea_eff = (
            self._kwargs.get("reasoning_effort", None)
            if reasoning_effort is None
            else reasoning_effort
        )
        if self._extra_body is not None:
            merged = deepcopy(self._extra_body)
            if extra_body is not None:
                merged.update(extra_body)
        else:
            merged = extra_body
        return ChatParameters(
            model=self._model,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
            max_retries=self._max_retries,
            base_url=self._base_url,
            temperature=temperature,
            image_detail=image_detail,
            reasoning_effort=rea_eff,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            extra_body=merged,
        )

    def complete(
        self,
        messages: Messages,
        response_type: type[ResponseType],
        chat_parameters: ChatParameters,
        *,
        base_delay: float,
        max_parse_retries: int,
        verbose: bool,
    ) -> LlmOutput[ResponseType]:
        """Issue one Chat Completions request for the given message list and
        return a populated :class:`LlmOutput`.

        This is the single place that talks to the SDK. Transport errors
        (rate-limit / 5xx / connection) are handled by the OpenAI client's own
        retry mechanism (configured via ``max_retries``); only *semantic*
        failures — an empty response or a structured-output parse error — are
        retried here, with exponential backoff. ``messages`` is read but never
        mutated, so callers own all conversation state. Stateless and
        thread-safe.
        """
        llm_output: Optional[LlmOutput[ResponseType]] = None
        for r in range(max_parse_retries + 1):
            start_time = time.perf_counter()
            llm_output = LlmOutput[response_type](parameters=chat_parameters)
            try:
                with _maybe_loading("Generating response...", disable=not verbose):
                    completion: ChatCompletion = self._client.chat.completions.create(
                        messages=messages.to_api_format(),
                        stream=False,
                        **chat_parameters.to_api_params(),
                    )
                if len(completion.choices) == 0:
                    raise LlmChatError("LLM returned 0 choice")
                choice = completion.choices[0]
                content = choice.message.content
                if content is None:
                    raise LlmChatError("LLM response is None")
                response = (
                    response_type.from_str(content)
                    if issubclass(response_type, ResponseModel)
                    else content
                )
                if chat_parameters.logprobs and (
                    choice.logprobs is None or choice.logprobs.content is None
                ):
                    raise LlmChatError("LLM logprobs is None")
            except (LlmChatError, ResponseModelParsingError) as e:
                # Semantic failures: retry locally up to max_parse_retries. The
                # backoff sleep is excluded from the recorded duration.
                llm_output.duration = max(time.perf_counter() - start_time, 1e-9)
                self._print_error(e, verbose)
                llm_output.detail = format_error(e)[0]
                if r == max_parse_retries:
                    raise
                time.sleep(base_delay**r + random.random())
                continue
            except Exception as e:
                # Everything else (incl. transport errors the SDK already
                # retried and gave up on) propagates immediately.
                llm_output.duration = max(time.perf_counter() - start_time, 1e-9)
                self._print_error(e, verbose)
                llm_output.detail = format_error(e)[0]
                raise
            else:
                llm_output.duration = max(time.perf_counter() - start_time, 1e-9)
                if verbose:
                    console.print(
                        Panel(
                            content,
                            title="LLM Response",
                            title_align="left",
                            border_style="green",
                        )
                    )
                llm_output.content = content
                llm_output.response = response
                llm_output.finish_reason = choice.finish_reason
                assert completion.usage is not None
                llm_output.prompt_tokens = completion.usage.prompt_tokens
                llm_output.completion_tokens = completion.usage.completion_tokens
                llm_output.cost = self._compute_cost(llm_output)
                llm_output.logprobs = (
                    choice.logprobs.content if choice.logprobs is not None else None
                )
                break

        assert llm_output is not None
        return llm_output

    def _compute_cost(self, llm_output: LlmOutput) -> float:
        cost = 0.0
        if (
            self._input_cost_per_token is not None
            and llm_output.prompt_tokens is not None
        ):
            cost += llm_output.prompt_tokens * self._input_cost_per_token
        if (
            self._output_cost_per_token is not None
            and llm_output.completion_tokens is not None
        ):
            cost += llm_output.completion_tokens * self._output_cost_per_token
        return self._add_cost(cost)

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def replicate(self) -> Self:
        return Llm(
            model=self._model,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
            max_retries=self._max_retries,
            api_key=self._api_key,
            base_url=self._base_url,
            input_cost=self._input_cost,
            output_cost=self._output_cost,
            extra_body=self._extra_body,
            **self._kwargs,
        )

    def _print_error(self, err: BaseException, verbose: bool) -> None:
        if not verbose:
            return
        console.print(
            Panel(
                format_error(err)[0],
                title="Error",
                title_align="left",
                border_style="red",
            )
        )


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("-t", "--temperature", type=float, default=None)
    parser.add_argument("--img", action="append", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    args = parser.parse_args()
    llm = Llm(args.model, api_key=args.api_key, base_url=args.base_url)
    llm_output = llm(args.prompt, temperature=args.temperature, images=args.img)
    print(llm_output.response)
