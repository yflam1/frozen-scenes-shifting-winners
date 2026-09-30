import threading
from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, Sequence, Union

from pydantic import Field, validate_call
from rich.panel import Panel

from .. import console
from ..dtypes import ImgLike, NonEmptyStr, SDict
from .msg import Messages
from .output import LlmOutput, ResponseType
from .template import PromptTemplate

if TYPE_CHECKING:  # avoid a circular import with ``llm.py`` at runtime
    from .llm import Llm


class Conversation:
    """A single, stateful multi-turn dialogue with one :class:`Llm` client.

    A conversation owns:

    * its **own system prompt**,
    * an ordered list of user/assistant **messages** (each user turn may carry
      images, if the model supports vision),
    * a per-turn **history** of :class:`LlmOutput` records.

    All mutating operations are guarded by a re-entrant lock, so a single
    conversation may be driven from multiple threads, and many *independent*
    conversations (each with its own lock) run safely in parallel — see
    :func:`concurrent_map`.

    A conversation can be **branched** at any point (:meth:`branch`): the copy
    shares the same client but gets an independent message history, so the two
    can diverge with different downstream prompts.
    """

    def __init__(
        self,
        client: "Llm",
        *,
        sys_prompt: Optional[NonEmptyStr] = None,
        verbose: bool = True,
        _messages: Optional[Messages] = None,
        _history: Optional[Sequence[LlmOutput]] = None,
    ) -> None:
        self._client = client
        self._verbose = verbose
        self._lock = threading.RLock()
        self._messages = _messages if _messages is not None else Messages()
        self._history: list[LlmOutput] = list(_history) if _history is not None else []
        if _messages is None and sys_prompt is not None:
            self._messages.set_system(sys_prompt)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def send(
        self,
        prompt: Union[NonEmptyStr, PromptTemplate],
        response_type: type[ResponseType] = str,
        *,
        images: Optional[Sequence[ImgLike]] = None,
        image_detail: Literal["auto", "low", "high"] = "auto",
        temperature: Optional[Annotated[float, Field(ge=0.0, le=2.0)]] = None,
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[Annotated[int, Field(ge=0, le=5)]] = None,
        extra_body: Optional[SDict[Any]] = None,
        base_delay: Annotated[float, Field(ge=1.0)] = 1.5,
        max_parse_retries: Annotated[int, Field(ge=0)] = 2,
    ) -> LlmOutput[ResponseType]:
        """Append a user turn, query the model, append the reply, and return
        the :class:`LlmOutput`.

        On any failure the appended user turn is rolled back so the
        conversation stays consistent and can be retried.
        """
        chat_parameters = self._client.build_params(
            temperature=temperature,
            image_detail=image_detail,
            reasoning_effort=reasoning_effort,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            extra_body=extra_body,
        )
        final_pmt = prompt() if isinstance(prompt, PromptTemplate) else prompt
        with self._lock:
            if self._verbose:
                console.print(
                    Panel(
                        final_pmt,
                        title="User Prompt",
                        title_align="left",
                        border_style="blue",
                    )
                )
            self._messages.add_user(final_pmt, images=images, image_detail=image_detail)
            try:
                llm_output = self._client.complete(
                    self._messages,
                    response_type,
                    chat_parameters,
                    base_delay=base_delay,
                    max_parse_retries=max_parse_retries,
                    verbose=self._verbose,
                )
            except Exception:
                # Roll back the user turn so the dialogue stays consistent.
                self._messages.remove_last()
                raise
            assert llm_output.content is not None
            self._messages.add_assistant(llm_output.content)
            llm_output.messages = self._messages.copy()
            self._history.append(llm_output)
            return llm_output

    def branch(self, at: Optional[int] = None) -> "Conversation":
        """Duplicate this conversation into an independent one.

        ``at`` is the number of leading messages to keep (the system message,
        if present, is index 0). With ``at=None`` the whole conversation is
        copied. The returned conversation shares the same client but has its
        own message history and lock, so the two can diverge freely.
        """
        if at is not None and at < 0:
            raise ValueError(f"`at` must be non-negative, got {at}")
        with self._lock:
            messages = self._messages.copy()
            if at is not None:
                if at > len(messages):
                    raise ValueError(
                        f"Cannot branch at {at}: conversation has "
                        f"{len(messages)} messages"
                    )
                messages.truncate(at)
                kept = sum(
                    1
                    for out in self._history
                    if out.messages is not None and len(out.messages) <= at
                )
                history = deepcopy(self._history[:kept])
            else:
                history = deepcopy(self._history)
            return Conversation(
                self._client,
                verbose=self._verbose,
                _messages=messages,
                _history=history,
            )

    def clear(self) -> None:
        """Drop all messages and history, keeping the system prompt."""
        with self._lock:
            system = self._messages.messages[0] if self._messages.has_system else None
            self._messages.clear()
            if system is not None:
                self._messages.set_system(system.content)
            self._history.clear()

    @validate_call
    def set_system(self, sys_prompt: NonEmptyStr) -> None:
        with self._lock:
            self._messages.set_system(sys_prompt)

    @property
    def messages(self) -> Messages:
        with self._lock:
            return self._messages.copy()

    @property
    def history(self) -> list[LlmOutput]:
        with self._lock:
            return deepcopy(self._history)

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)
