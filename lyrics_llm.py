import re
import requests
from openai import OpenAI
from prompt_templates import PromptBuilder


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
        pass

    # Fallback to /api/generate
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


def _openai_generate(prompt, system, api_key, base_url, model, temperature=0.7, timeout=120):
    """Call OpenAI-compatible API."""
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



def generate_checked_fields(description, title, lyrics, tags,
                            gen_title=True, gen_lyrics=True, gen_tags=True,
                            backend="ollama", max_length_sec=None, **kwargs):
    """Generate multiple fields in a single LLM call.

    Fields that are not checked for generation but have values are provided
    as context to the LLM. Returns dict with keys 'title', 'lyrics', 'tags'
    containing either the generated or original values.
    """
    fields_to_generate = []
    if gen_title:
        fields_to_generate.append("title")
    if gen_lyrics:
        fields_to_generate.append("lyrics")
    if gen_tags:
        fields_to_generate.append("tags")

    if not fields_to_generate:
        return {"title": title, "lyrics": lyrics, "tags": tags}

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

    result = {"title": title, "lyrics": lyrics, "tags": tags}

    # Parse response
    if gen_title:
        m = re.search(r'===TITLE===\s*(.*?)\s*===END_TITLE===', response, re.DOTALL)
        if m:
            result["title"] = m.group(1).strip()

    if gen_lyrics:
        m = re.search(r'===LYRICS===\s*(.*?)\s*===END_LYRICS===', response, re.DOTALL)
        if m:
            result["lyrics"] = m.group(1).strip()

    if gen_tags:
        m = re.search(r'===TAGS===\s*(.*?)\s*===END_TAGS===', response, re.DOTALL)
        if m:
            result["tags"] = m.group(1).strip()

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
    except Exception as e:
        return []


def generate_with_ollama(backend, system_prompt, user_prompt, **kwargs):
    """Generic LLM generation with custom system and user prompts."""
    return _call_llm(user_prompt, system_prompt, backend, **kwargs)
