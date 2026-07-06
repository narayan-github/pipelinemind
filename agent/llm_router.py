"""
LLM Router — routes each call type to the cheapest/fastest model tier.

Model update (2024-05-14):
  llama3-8b-8192   → DECOMMISSIONED by Groq
  llama3-70b-8192  → DECOMMISSIONED by Groq

Current model assignments:
  SUMMARY / INTENT / HYDE → llama-3.1-8b-instant  (fast, cheap, good for structured tasks)
  AGENT                   → llama-3.3-70b-versatile (function-calling, reasoning)

Rate-limit strategy:
  - By routing INTENT + HYDE to 8b-instant, ~70% of the 70b quota is preserved
    for agent function-calling loops which actually need the large model.
  - Secondary key rotation: set GROQ_API_KEY_SECONDARY in .env to double quota.
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from functools import lru_cache
from typing import Any

from groq import Groq, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pm_config import settings

logger = logging.getLogger(__name__)


class CallType(str, Enum):
    SUMMARY = "summary"
    INTENT  = "intent"
    HYDE    = "hyde"
    AGENT   = "agent"


# Model assignment — reads from settings which reads from .env
# This means you can override any model via .env without code changes.
def _build_model_map() -> dict[CallType, str]:
    return {
        CallType.SUMMARY: settings.groq_model_fast,    # llama-3.1-8b-instant
        CallType.INTENT:  settings.groq_model_fast,    # llama-3.1-8b-instant
        CallType.HYDE:    settings.groq_model_fast,    # llama-3.1-8b-instant
        CallType.AGENT:   settings.groq_model_agent,   # llama-3.3-70b-versatile
    }


_MAX_TOKENS: dict[CallType, int] = {
    CallType.SUMMARY: 200,
    CallType.INTENT:  60,
    CallType.HYDE:    250,
    CallType.AGENT:   2048,
}

_TEMPERATURE: dict[CallType, float] = {
    CallType.SUMMARY: 0.1,
    CallType.INTENT:  0.0,
    CallType.HYDE:    0.35,
    CallType.AGENT:   0.2,
}


@lru_cache(maxsize=8)
def _get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def _primary_client() -> Groq:
    return _get_client(settings.groq_api_key)


def _secondary_client() -> Groq | None:
    secondary = os.environ.get("GROQ_API_KEY_SECONDARY", "").strip()
    if secondary and secondary != settings.groq_api_key:
        return _get_client(secondary)
    return None


class LLMRouter:
    """
    Routes LLM calls to the appropriate model tier.
    Implements key-rotation on 429 if a secondary key is configured.
    Logs model usage statistics for debugging.
    """

    def __init__(self) -> None:
        self._call_counts: dict[str, int] = {}
        self._rate_limit_hits = 0
        self._model_errors: dict[str, int] = {}

    def complete(
        self,
        call_type: CallType,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        extra_kwargs: dict | None = None,
    ) -> Any:
        model_map   = _build_model_map()
        model       = model_map[call_type]
        max_tokens  = _MAX_TOKENS[call_type]
        temperature = _TEMPERATURE[call_type]

        kwargs: dict = {
            "model":       model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"]       = tools
            kwargs["tool_choice"] = tool_choice
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        self._call_counts[call_type.value] = (
            self._call_counts.get(call_type.value, 0) + 1
        )
        logger.debug("LLMRouter: %s → %s", call_type.value, model)
        return self._call_with_fallback(kwargs)

    def _call_with_fallback(self, kwargs: dict) -> Any:
        try:
            return _primary_client().chat.completions.create(**kwargs)
        except RateLimitError as exc:
            self._rate_limit_hits += 1
            logger.warning(
                "Groq 429 on primary key (total_hits=%d) model=%s",
                self._rate_limit_hits, kwargs.get("model"),
            )
            secondary = _secondary_client()
            if secondary:
                logger.info("Retrying with secondary Groq key")
                try:
                    return secondary.chat.completions.create(**kwargs)
                except RateLimitError:
                    logger.warning("Secondary key also rate-limited")
            raise
        except Exception as exc:
            model = kwargs.get("model", "unknown")
            self._model_errors[model] = self._model_errors.get(model, 0) + 1
            logger.error("LLMRouter error for model %s: %s", model, exc)
            raise

    def stats(self) -> dict:
        return {
            "call_counts":      self._call_counts,
            "rate_limit_hits":  self._rate_limit_hits,
            "model_errors":     self._model_errors,
            "model_assignment": {
                k.value: v for k, v in _build_model_map().items()
            },
        }


router = LLMRouter()
