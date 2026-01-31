"""LLM backend implementations for lyrics/tags/title generation."""

import logging

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)


class OllamaBackend:
    """Ollama API backend."""

    def __init__(self, base_url, model, temperature=0.7):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

    def generate(self, prompt, system):
        options = {"temperature": self.temperature}
        # Try /api/chat first
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": options,
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.HTTPError:
            logger.warning("Ollama /api/chat failed, falling back to /api/generate")

        # Fallback to /api/generate
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": options,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def unload(self):
        """Request Ollama to free VRAM for this model."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=30,
            )
            resp.raise_for_status()
            return f"Unloaded {self.model} from Ollama."
        except Exception as e:
            logger.warning("Failed to unload Ollama model %s: %s", self.model, e)
            return f"Error unloading {self.model}: {e}"

    @staticmethod
    def list_models(base_url="http://localhost:11434"):
        """List available Ollama models."""
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
        except Exception as e:
            logger.warning("Failed to list Ollama models: %s", e)
            return []


class OpenAIBackend:
    """OpenAI-compatible API backend."""

    def __init__(self, api_key, base_url, model, temperature=0.7):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt, system):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content


def clean_llm_response(text):
    """Strip whitespace and remove markdown code block wrappers if present."""
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()
    return text
