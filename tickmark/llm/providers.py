"""OpenAI and Anthropic adapters behind the ChatModel interface.

Provider exceptions are converted to LLMError so the engine can degrade gracefully
without knowing which SDK raised.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from tickmark.llm.base import Completion, LLMError, StreamResult
from tickmark.llm.pricing import usage_for

REQUEST_TIMEOUT_S = 60.0


class OpenAIChat:
    provider = "openai"

    def __init__(self, model: str, api_key: str | None = None):
        import openai

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set")
        self._openai = openai
        self.client = openai.OpenAI(api_key=key, timeout=REQUEST_TIMEOUT_S, max_retries=2)
        self.model = model

    def complete(self, system: str, user: str, *, max_tokens: int = 800, json_mode: bool = False) -> Completion:
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_completion_tokens=max_tokens,
                **kwargs,
            )
        except self._openai.OpenAIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        usage = response.usage
        return Completion(
            text=response.choices[0].message.content or "",
            usage=usage_for(self.model, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0),
        )

    def stream(self, system: str, user: str, *, max_tokens: int = 1500) -> StreamResult:
        result: StreamResult

        def deltas() -> Iterator[str]:
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_completion_tokens=max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                for chunk in stream:
                    if chunk.usage:
                        result.usage = usage_for(self.model, chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except self._openai.OpenAIError as exc:
                raise LLMError(f"OpenAI stream failed: {exc}") from exc

        result = StreamResult(deltas())
        return result


class AnthropicChat:
    provider = "anthropic"

    def __init__(self, model: str, api_key: str | None = None):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=key, timeout=REQUEST_TIMEOUT_S, max_retries=2)
        self.model = model

    def complete(self, system: str, user: str, *, max_tokens: int = 800, json_mode: bool = False) -> Completion:
        if json_mode:
            system += "\n\nRespond with a single JSON object and nothing else."
        try:
            message = self.client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
            )
        except self._anthropic.AnthropicError as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        text = "".join(block.text for block in message.content if block.type == "text")
        return Completion(text, usage_for(self.model, message.usage.input_tokens, message.usage.output_tokens))

    def stream(self, system: str, user: str, *, max_tokens: int = 1500) -> StreamResult:
        result: StreamResult

        def deltas() -> Iterator[str]:
            try:
                with self.client.messages.stream(
                    model=self.model,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=max_tokens,
                ) as stream:
                    yield from stream.text_stream
                    final = stream.get_final_message()
                    result.usage = usage_for(self.model, final.usage.input_tokens, final.usage.output_tokens)
            except self._anthropic.AnthropicError as exc:
                raise LLMError(f"Anthropic stream failed: {exc}") from exc

        result = StreamResult(deltas())
        return result


def make_chat_model(provider: str, model: str):
    if provider == "openai":
        return OpenAIChat(model)
    if provider == "anthropic":
        return AnthropicChat(model)
    raise ValueError(f"No chat model for provider {provider!r}")
