"""
Unified LLM client — Stage 5/6.

All LLM calls are routed through this module so the model (Claude / GPT-4o)
can be swapped in pipeline.yaml without touching prompt logic. Temperature is
kept at 0.3 by default — we want grounded, mechanistically plausible hypotheses,
not creative extrapolation. The model can be upgraded without changing any
prompt or parser code.
"""

import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class LLMClient:
    """Provider-agnostic wrapper for Claude and OpenAI completions."""

    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None

    @classmethod
    def from_config(cls, config: dict) -> "LLMClient":
        llm_cfg = config["llm"]
        return cls(
            model=llm_cfg["model"],
            temperature=llm_cfg["temperature"],
            max_tokens=llm_cfg["max_tokens"],
        )

    def _init_client(self) -> None:
        """Lazy-initialise the provider SDK based on model name prefix."""
        if "claude" in self.model:
            import anthropic
            self._client = anthropic.Anthropic()
        elif "gpt" in self.model:
            import openai
            self._client = openai.OpenAI()
        else:
            raise ValueError(f"Unrecognised model prefix (expected 'claude' or 'gpt'): {self.model}")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat completion request and return the response text.

        Raises RuntimeError on API errors after 3 retries with exponential backoff.
        All calls are logged at DEBUG level with token counts.
        """
        import time

        if self._client is None:
            self._init_client()

        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(3):
            try:
                if "claude" in self.model:
                    msg = self._client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    text = msg.content[0].text
                    logger.debug(
                        "Claude call: in=%d out=%d",
                        msg.usage.input_tokens,
                        msg.usage.output_tokens,
                    )
                elif "gpt" in self.model:
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    text = resp.choices[0].message.content
                    usage = resp.usage
                    logger.debug(
                        "OpenAI call: in=%d out=%d",
                        usage.prompt_tokens,
                        usage.completion_tokens,
                    )
                return text
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning("LLM attempt %d failed (%s); retrying in %ds", attempt + 1, exc, wait)
                time.sleep(wait)

        raise RuntimeError(f"LLM call failed after 3 attempts: {last_exc}") from last_exc
