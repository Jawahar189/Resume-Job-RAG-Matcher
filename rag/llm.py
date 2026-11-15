"""OpenAI-compatible chat client with schema-validated JSON output.

Works with OpenAI, Groq, Together, OpenRouter, a local Ollama server, etc. -
anything that speaks the OpenAI chat-completions API. Set LLM_BASE_URL to switch.
"""

import json
import re
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from . import config

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class LLMClient:
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)

    @classmethod
    def from_env(cls) -> "LLMClient | None":
        if not config.LLM_API_KEY and not config.LLM_BASE_URL:
            return None
        return cls(config.LLM_API_KEY, config.LLM_MODEL, config.LLM_BASE_URL)

    def structured(self, system: str, user: str, schema: type[T], max_attempts: int = 2) -> T:
        """Ask for JSON matching `schema`; on validation failure, feed the error back and retry."""
        schema_json = json.dumps(schema.model_json_schema())
        messages = [
            {
                "role": "system",
                "content": f"{system}\n\nRespond with a single JSON object that conforms to this JSON Schema "
                f"(no markdown, no commentary):\n{schema_json}",
            },
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        for _ in range(max_attempts):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = _FENCE.sub("", (response.choices[0].message.content or "").strip())
            try:
                return schema.model_validate_json(content)
            except ValidationError as exc:
                last_error = exc
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {"role": "user", "content": f"That JSON did not match the schema:\n{exc}\nReturn corrected JSON only."}
                )
        raise RuntimeError(f"LLM did not return valid {schema.__name__} JSON: {last_error}")
