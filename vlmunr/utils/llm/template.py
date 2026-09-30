from typing import Any, Optional

from pydantic import validate_call
from typing_extensions import Self

from ..dtypes import NonEmptyStr

DEFAULT_COSTAR_JSON_STYLE = """\
You must respond using only JSON without preamble, postamble, additional commentary, and Markdown formatting.
Do not miss any fields or add extra fields.
Your JSON must be complete (i.e., from a proper start with "{" to a proper end with "}") and well-formatted (e.g., strings are enclosed by double quotation marks and there are no newlines within strings)."""

DEFAULT_COSTAR_JSON_TONE = "Neutral"

DEFAULT_COSTAR_JSON_AUDIENCE = """\
A computer program that will strictly parse your JSON response."""


class PromptTemplate(object):
    @validate_call
    def __init__(self, template: NonEmptyStr, **variables: Any) -> None:
        """
        Initialize a PromptTemplate instance.

        Args:
            template (NonEmptyStr): A string whose instances of "{...}"
                will be substituted accordingly upon invocation of the
                `__call__()` method. If you want to escape some pairs of
                "{...}", use "{{...}}" instead.
            **variables (Any, optional): Names of placeholders which
                will be substituted upon invocation of the `__call__()`
                method. Defaults to None.

                If `template` contains "{year}", and you want to give it
                a default value of `2024`, then you should pass:

                ```python
                template = PromptTemplate("The year is {year}.", year=2025)
                ```

                But if "{year}" is not optional, then you must use
                `None`:

                ```python
                template = PromptTemplate("My name is {name}.", name=None)
                ```
        """
        self._template = template
        self._variables = self._process_substitutions(variables)

    @validate_call
    def __call__(self, **substitutions: Any) -> str:
        """
        Format the stored template using suitable substitutions.

        Args:
            **substitutions (Any, optional): Values that must be or are
                desired to be substituted into `self.template`. Defaults
                to None.

        Raises:
            ValueError: If `substitutions` does not contain required keys
                (i.e., keys in `self.variables` whose value is `None`).

        Returns:
            str: Formatted prompt.
        """
        substitutions = {
            **self._variables,
            **self._process_substitutions(substitutions),
        }
        omitted = {
            k
            for k, v in substitutions.items()
            if k in self._variables and v is None
        }
        if len(omitted) > 0:
            raise ValueError(f"Values of keys {omitted} must not be None")
        return self._template.format(**substitutions)

    @property
    def template(self) -> NonEmptyStr:
        return self._template

    @property
    def variables(self) -> dict[NonEmptyStr, Optional[str]]:
        return self._variables

    @validate_call
    def _process_substitutions(
        self, sub: dict[NonEmptyStr, Any]
    ) -> dict[NonEmptyStr, Optional[str]]:
        return {k: None if v is None else str(v) for k, v in sub.items()}


class CoStar(PromptTemplate):
    _TEMPLATE = """\
## Context
{context}

## Objective
{objective}

## Style
{style}

## Tone
{tone}

## Audience
{audience}

## Response
{response}
"""

    @validate_call
    def __init__(
        self,
        context: Optional[str] = None,
        objective: Optional[str] = None,
        style: Optional[str] = None,
        tone: Optional[str] = None,
        audience: Optional[str] = None,
        response: Optional[str] = None,
    ) -> None:
        super().__init__(
            self._TEMPLATE,
            context=context,
            objective=objective,
            style=style,
            tone=tone,
            audience=audience,
            response=response,
        )

    @validate_call
    def __call__(self, **kwargs: Optional[str]) -> str:
        substitutions = {k: v for k, v in kwargs.items() if v is not None}
        return super().__call__(**substitutions)

    @classmethod
    @validate_call
    def Json(
        cls,
        context: Optional[str] = None,
        objective: Optional[str] = None,
        style: Optional[str] = DEFAULT_COSTAR_JSON_STYLE,
        tone: Optional[str] = DEFAULT_COSTAR_JSON_TONE,
        audience: Optional[str] = DEFAULT_COSTAR_JSON_AUDIENCE,
        response: Optional[str] = None,
    ) -> Self:
        return cls(
            context=context,
            objective=objective,
            style=style,
            tone=tone,
            audience=audience,
            response=response,
        )
