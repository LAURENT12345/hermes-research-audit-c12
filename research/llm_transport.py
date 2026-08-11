"""LLM transport boundary for the research reasoner adapter (Commit 12).

The research domain never talks to a vendor directly. This module
defines the smallest transport interface the adapter needs::

    LLMReasoner -> LLMTransport.complete(messages) -> raw response text

Only raw text crosses this boundary. Structured-output parsing lives in
``research.llm_reasoner``; every authoritative decision stays in the
deterministic verifier / domain layers.

Transport convention follows the repository: OpenCode Zen / OpenCode Go
are OpenAI-compatible endpoints reached through the ``openai`` SDK (a
hard dependency of this project). Credentials come exclusively from
environment variables using the repository's canonical names
(``OPENCODE_GO_API_KEY`` / ``OPENCODE_GO_BASE_URL``,
``OPENCODE_ZEN_API_KEY`` / ``OPENCODE_ZEN_BASE_URL``). Nothing is
hard-coded and nothing is ever committed.

This module is deliberately free of any ``research.*`` import: the
transport does not know about claims, evidence, proposals or the store.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMTransportError(Exception):
    """Base class for transport-level failures (connection, auth, 5xx)."""


class LLMTimeoutError(LLMTransportError):
    """The model did not answer within the configured timeout."""


# ---------------------------------------------------------------------------
# Provider defaults (endpoints mirror hermes_cli/auth.py ProviderConfig)
# ---------------------------------------------------------------------------

#: provider -> (api-key env var, base-url env var, default endpoint)
_PROVIDER_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "opencode-go": (
        "OPENCODE_GO_API_KEY",
        "OPENCODE_GO_BASE_URL",
        "https://opencode.ai/zen/go/v1",
    ),
    "opencode-zen": (
        "OPENCODE_ZEN_API_KEY",
        "OPENCODE_ZEN_BASE_URL",
        "https://opencode.ai/zen/v1",
    ),
}


@dataclass(frozen=True)
class TransportConfig:
    """Resolved transport configuration.

    ``api_key`` and ``base_url`` may be empty; the transport factory
    validates that a real call is possible before constructing a client.
    """

    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""
    timeout: float = 60.0
    #: Extra kwargs forwarded to the OpenAI SDK client (kept for
    #: test/override flexibility; not part of the public contract).
    client_kwargs: dict = field(default_factory=dict)


def resolve_transport_config(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
) -> TransportConfig:
    """Resolve a config from explicit arguments, environment, and defaults.

    Resolution order: explicit argument > environment variable > default.
    The API key is read ONLY from the environment (never from arguments
    that could leak into logs or git history).
    """
    provider = (provider or "opencode-go").strip()
    key_env, url_env, default_url = _PROVIDER_DEFAULTS.get(
        provider, (None, None, "")
    )
    resolved_key = api_key or (
        os.environ.get(key_env, "") if key_env else ""
    )
    resolved_url = base_url or (
        os.environ.get(url_env) if url_env else ""
    ) or default_url
    return TransportConfig(
        provider=provider,
        model=(model or "").strip(),
        base_url=resolved_url,
        api_key=resolved_key,
        timeout=float(timeout or 60.0),
    )


# ---------------------------------------------------------------------------
# Transport protocol + OpenAI-compatible implementation
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMTransport(Protocol):
    """The smallest transport surface the adapter consumes.

    ``messages`` is a sequence of ``{"role": ..., "content": ...}`` dicts
    (the chat-completions shape). The implementation returns the raw
    assistant text; any non-text failure is raised as
    :class:`LLMTransportError` (or :class:`LLMTimeoutError`).
    """

    def complete(
        self, messages: Sequence[dict], *, timeout: float
    ) -> str:
        """Return the raw assistant message content."""


class ChatCompletionsTransport:
    """OpenAI-compatible chat-completions transport (OpenCode Zen / Go).

    The ``openai`` SDK is imported lazily so that importing this module
    (and thus the research adapter) never requires the SDK at import
    time and never touches the network.
    """

    def __init__(self, config: TransportConfig):
        if not config.model:
            raise LLMTransportError(
                "no model configured for LLM reasoner transport"
            )
        if not config.api_key:
            key_env = (
                _PROVIDER_DEFAULTS.get(config.provider, (None,))[0]
                or "the provider API key env var"
            )
            raise LLMTransportError(
                f"no API key for provider {config.provider!r}; set {key_env}"
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or None,
            timeout=config.timeout,
            **config.client_kwargs,
        )
        self._model = config.model

    def complete(
        self, messages: Sequence[dict], *, timeout: float
    ) -> str:
        import openai

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=list(messages),
                timeout=timeout,
            )
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError(
                f"model {self._model} timed out after {timeout}s"
            ) from exc
        except openai.APIConnectionError as exc:
            raise LLMTransportError(
                f"cannot reach {self._model} endpoint: {exc}"
            ) from exc
        except openai.APIError as exc:
            raise LLMTransportError(
                f"{type(exc).__name__} from {self._model}: {exc}"
            ) from exc
        try:
            content = response.choices[0].message.content
        except (IndexError, AttributeError) as exc:
            raise LLMTransportError(
                "model response had no usable message content"
            ) from exc
        return content or ""


def build_transport(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[TransportConfig] = None,
) -> ChatCompletionsTransport:
    """Resolve configuration and build a ready-to-use transport."""
    resolved = config or resolve_transport_config(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )
    return ChatCompletionsTransport(resolved)
