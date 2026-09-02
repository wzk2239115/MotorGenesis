"""LLM agent backend: GLM-5.3 via the 360 Proxy OpenAI-compatible endpoint.

Reads credentials from the opencode config (``~/.config/opencode/opencode.json``)
so the loop reuses the same API opencode itself uses, with no extra setup.
GLM-5.3 is a reasoning model: the answer is in ``content``, the chain-of-thought
in ``reasoning_content``; we give a generous ``max_tokens`` so reasoning does
not eat the answer budget.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def load_opencode_config() -> dict:
    """Return ``{base_url, api_key, model}`` from the opencode config.

    Falls back to environment variables ``MOTORGENESIS_LLM_*`` if the config
    is absent, then to sensible defaults for the 360 Proxy.
    """
    import os

    candidates = [
        Path.home() / ".config" / "opencode" / "opencode.json",
        Path.home() / ".config" / "opencode" / "opencode.jsonc",
    ]
    for path in candidates:
        if path.is_file():
            try:
                cfg = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            provider = cfg.get("provider", {})
            proxy = provider.get("360-proxy", {})
            options = proxy.get("options", {})
            base_url = options.get("baseURL")
            api_key = options.get("apiKey")
            models = proxy.get("models", {})
            # Prefer the configured default model, else the first model listed.
            model = cfg.get("model") or ""
            if model and "/" in model and model.split("/", 1)[0] not in models:
                model = model.split("/", 1)[1]
            if not model and models:
                model = next(iter(models))
            if base_url and api_key:
                return {"base_url": base_url, "api_key": api_key, "model": model or "z-ai/glm-5.3"}
    return {
        "base_url": os.environ.get("MOTORGENESIS_LLM_BASE_URL", "https://api.360.cn/v1"),
        "api_key": os.environ.get("MOTORGENESIS_LLM_API_KEY", ""),
        "model": os.environ.get("MOTORGENESIS_LLM_MODEL", "z-ai/glm-5.3"),
    }


@dataclass
class LLMResponse:
    content: str
    reasoning: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient:
    """Thin OpenAI-compatible wrapper around the configured model."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None, timeout: float = 180.0):
        config = load_opencode_config()
        self.base_url = base_url or config["base_url"]
        self.api_key = api_key or config["api_key"]
        self.model = model or config["model"]
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "No LLM API key found. Set MOTORGENESIS_LLM_API_KEY or configure "
                "~/.config/opencode/opencode.json with a 360-proxy provider."
            )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import httpx
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                http_client=httpx.Client(timeout=self.timeout),
            )
        return self._client

    def complete(self, messages: Iterable[dict], *, max_tokens: int = 8192, temperature: float = 0.4) -> LLMResponse:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        msg = r.choices[0].message
        usage = r.usage
        return LLMResponse(
            content=msg.content or "",
            reasoning=getattr(msg, "reasoning_content", "") or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    """Pull the first ```python``` block out of an LLM reply."""
    if not text:
        return None
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    # Some models emit raw code with no fence; accept it if it looks like Python.
    if "def build" in text:
        return text.strip()
    return None
