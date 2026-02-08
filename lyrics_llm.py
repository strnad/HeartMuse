import logging
import re
import requests
from openai import OpenAI
from prompt_templates import PromptBuilder

logger = logging.getLogger(__name__)


def _ollama_generate(prompt, system, base_url, model, temperature=0.7, timeout=120):
    """Call Ollama API, trying /api/chat first, falling back to /api/generate."""
    options = {"temperature": temperature}
    # Try /api/chat first
    try:
        resp = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": options,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.HTTPError:
        logger.info("Ollama /api/chat failed, falling back to /api/generate")
    except requests.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {base_url}. "
            "Is Ollama running? Start it with 'ollama serve'."
        )
    except requests.Timeout:
        raise TimeoutError(
            f"Ollama request timed out after {timeout}s. "
            "Try increasing the timeout or using a smaller model."
        )

    # Fallback to /api/generate
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": options,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]
    except requests.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {base_url}. "
            "Is Ollama running? Start it with 'ollama serve'."
        )
    except requests.Timeout:
        raise TimeoutError(
            f"Ollama request timed out after {timeout}s. "
            "Try increasing the timeout or using a smaller model."
        )


def _openai_generate(prompt, system, api_key, base_url, model, temperature=0.7, timeout=120):
    """Call OpenAI-compatible API."""
    if not api_key:
        raise ValueError("OpenAI API key is not configured. Set OPENAI_API_KEY in your .env file.")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def _call_llm(prompt, system, backend, **kwargs):
    temperature = kwargs.get("temperature", 0.7)
    timeout = kwargs.get("timeout", 120)
    if backend == "ollama":
        return _ollama_generate(
            prompt, system,
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            model=kwargs.get("model", "llama3"),
            temperature=temperature,
            timeout=timeout,
        )
    elif backend == "openai":
        return _openai_generate(
            prompt, system,
            api_key=kwargs.get("api_key", ""),
            base_url=kwargs.get("base_url", "https://api.openai.com/v1"),
            model=kwargs.get("model", "gpt-4o-mini"),
            temperature=temperature,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


_FIELD_PATTERNS = {
    "description": r'===DESCRIPTION===\s*(.*?)\s*===END_DESCRIPTION===',
    "title": r'===TITLE===\s*(.*?)\s*===END_TITLE===',
    "lyrics": r'===LYRICS===\s*(.*?)\s*===END_LYRICS===',
    "tags": r'===TAGS===\s*(.*?)\s*===END_TAGS===',
}


def _normalize_tags(tags_str: str) -> str:
    """Normalize comma-separated tags: lowercase, trim, deduplicate, remove empties."""
    tags = [t.strip().lower() for t in tags_str.split(",")]
    seen = set()
    result = []
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
    return ",".join(result)


def generate_checked_fields(description, title, lyrics, tags,
                            gen_desc=False, gen_title=True, gen_lyrics=True, gen_tags=True,
                            backend="ollama", max_length_sec=None, **kwargs):
    """Generate multiple fields in a single LLM call.

    Fields that are not checked for generation but have values are provided
    as context to the LLM. Returns dict with keys 'description', 'title',
    'lyrics', 'tags' containing either the generated or original values.
    Also returns 'failed_fields' list with names of fields that failed to parse.
    """
    fields_to_generate = []
    if gen_desc:
        fields_to_generate.append("description")
    if gen_title:
        fields_to_generate.append("title")
    if gen_lyrics:
        fields_to_generate.append("lyrics")
    if gen_tags:
        fields_to_generate.append("tags")

    if not fields_to_generate:
        return {"description": description, "title": title, "lyrics": lyrics, "tags": tags, "failed_fields": []}

    # Build context dict for PromptBuilder
    context = {
        "description": description,
        "title": title,
        "lyrics": lyrics,
        "tags": tags,
    }

    # Build prompts using PromptBuilder
    system_prompt = PromptBuilder.build_system_prompt(fields_to_generate)
    user_prompt = PromptBuilder.build_user_prompt(fields_to_generate, context, max_length_sec)

    response = _call_llm(user_prompt, system_prompt, backend, **kwargs)

    result = {"description": description, "title": title, "lyrics": lyrics, "tags": tags, "failed_fields": []}

    # Parse response for each requested field
    for field in fields_to_generate:
        pattern = _FIELD_PATTERNS[field]
        m = re.search(pattern, response, re.DOTALL)
        if m:
            value = m.group(1).strip()
            if field == "tags":
                value = _normalize_tags(value)
            result[field] = value
        else:
            logger.warning("Failed to parse %s from LLM response (markers not found)", field)
            result["failed_fields"].append(field)

    return result


def unload_ollama_model(base_url="http://localhost:11434", model="llama3"):
    """Unload an Ollama model from VRAM."""
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
        resp.raise_for_status()
        return f"Unloaded {model} from Ollama."
    except Exception as e:
        return f"Error unloading {model}: {e}"


def list_ollama_models(base_url="http://localhost:11434"):
    """List available Ollama models."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [m["name"] for m in models]
    except Exception:
        logger.warning("Failed to list Ollama models at %s", base_url)
        return []
