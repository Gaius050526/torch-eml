"""LLM trunk adapter for extracting numerical features from text.

Provides a thin wrapper around LLM APIs (Anthropic or OpenAI) that
prompts the model to extract structured numerical features from
unstructured text input.
"""

from __future__ import annotations

import json
from typing import Sequence

import torch


def _call_anthropic(model: str, system: str, user_text: str) -> str:
    """Call Anthropic API and return text response."""
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_text}],
    )
    return msg.content[0].text


def _call_openai(model: str, system: str, user_text: str) -> str:
    """Call OpenAI API and return text response."""
    import openai

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        max_tokens=1024,
    )
    return resp.choices[0].message.content


class LLMTrunk:
    """Thin wrapper that turns an LLM API into a numerical feature extractor.

    Args:
        provider: "anthropic" or "openai".
        model: Model name (e.g. "claude-opus-4-6", "gpt-4o").
        features: List of dicts with "name" and "description" keys.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        features: Sequence[dict[str, str]],
    ):
        if provider not in ("anthropic", "openai"):
            raise ValueError(f"provider must be 'anthropic' or 'openai', got '{provider}'")
        self.provider = provider
        self.model = model
        self.features = list(features)
        self.feature_names = [f["name"] for f in self.features]
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        feature_lines = []
        for f in self.features:
            feature_lines.append(f'  - "{f["name"]}": {f["description"]}')
        features_str = "\n".join(feature_lines)

        return (
            "You are a numerical feature extractor. Given the user's text, "
            "extract the following features as numbers. Respond with ONLY a "
            "JSON object, no other text.\n\n"
            f"Features to extract:\n{features_str}\n\n"
            "Respond with a JSON object where each key is a feature name "
            "and each value is a number (int or float). No nulls, no strings, "
            "no explanations."
        )

    def _call_llm(self, text: str) -> str:
        if self.provider == "anthropic":
            return _call_anthropic(self.model, self._system_prompt, text)
        else:
            return _call_openai(self.model, self._system_prompt, text)

    def _parse_response(self, response: str) -> dict[str, float]:
        """Parse JSON response and validate all features are present and numeric."""
        data = json.loads(response)

        for name in self.feature_names:
            if name not in data:
                raise ValueError(f"Missing feature in response: {name}")
            if not isinstance(data[name], (int, float)):
                raise ValueError(
                    f"Feature '{name}' is not numeric: {data[name]!r}"
                )

        return {name: float(data[name]) for name in self.feature_names}

    def extract(self, text: str) -> torch.Tensor:
        """Extract features from text using the LLM.

        Args:
            text: Input text to extract features from.

        Returns:
            Tensor of shape [1, n_features].

        Raises:
            ValueError: If features are missing or non-numeric after retry.
        """
        last_error = None
        for attempt in range(2):
            try:
                response = self._call_llm(text)
                parsed = self._parse_response(response)
                values = [parsed[name] for name in self.feature_names]
                return torch.tensor([values], dtype=torch.float32)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                continue

        raise ValueError(f"Failed to extract features after retry: {last_error}")
