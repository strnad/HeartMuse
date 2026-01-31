import re
import requests
from openai import OpenAI

GENERATE_ALL_SYSTEM_PROMPT = """You are an expert songwriter and music producer. You will be given context about a song and asked to generate specific fields.

IMPORTANT: Think step by step to ensure all fields are coherent with each other:
1. First, come up with the TITLE — it should capture the core theme/emotion of the song description.
2. Then, write the LYRICS — they must reflect the title and the description. The title should feel like a natural fit for the lyrics.
3. Finally, derive the TAGS — base them on the actual musical qualities of the lyrics and title you just wrote, not just the description.

If only some fields are requested, still consider the provided context (existing title, lyrics, tags) to keep everything consistent.

You MUST output EXACTLY the requested fields in this format (use these exact markers):

===TITLE===
song title here (1-5 words, no quotes)
===END_TITLE===

===LYRICS===
[verse]
lyrics here

[chorus]
lyrics here
===END_LYRICS===

===TAGS===
comma,separated,tags,here
===END_TAGS===

Only output the fields that are requested. Do not output fields that are not requested.
For lyrics, use sections like [verse], [chorus], [bridge], [outro], [intro]. Keep lyrics natural and singable.
For tags, use comma-separated MUSIC PRODUCTION tags ONLY. Each tag is a single word or a short multi-word phrase. Use 5-15 tags total. Choose from these categories:
- Genre/subgenre: pop, rock, jazz, classical, electronic, blues, country, funk, reggae, hip-hop, rap, indie, folk, soul, r&b, latin, dance, disco, techno, house, ambient, lo-fi, hard rock, progressive rock, dream pop, synth-pop, bossa nova
- Mood/emotion: happy, sad, romantic, energetic, calm, peaceful, melancholic, upbeat, chill, dark, bright, dreamy, aggressive, gentle, intense, relaxing, cheerful, nostalgic, dramatic, playful
- Instruments: piano, guitar, bass, drum, violin, saxophone, trumpet, flute, organ, synthesizer, acoustic guitar, electric guitar, cello, harp, strings, brass
- Vocal type: male voice, female voice, choir, vocal harmony, rap, duet (MUST match the song description)
- Tempo/style: fast, slow, medium tempo, groovy, driving, flowing, bouncy, acoustic, orchestral, cinematic, atmospheric, minimal, lo-fi, polished
Do NOT include content/story/character/theme tags (e.g. no "love", "heartbreak", "summer" as tags). Focus strictly on how the music SOUNDS, not what it is about.
For title, keep it 1-5 words, creative and fitting."""


def _add_dots_to_abbreviations(text):
    """Convert uppercase abbreviations to dotted form (e.g. AI -> A.I., IT -> I.T.)
    so that the music generator reads them as individual letters."""
    def _replace(m):
        word = m.group(0)
        return ".".join(word) + "."
    return re.sub(r'\b[A-Z]{2,}\b', _replace, text)


def _ollama_generate(prompt, system, base_url, model, temperature=0.7):
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
            timeout=120,
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
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _openai_generate(prompt, system, api_key, base_url, model, temperature=0.7):
    """Call OpenAI-compatible API."""
    client = OpenAI(api_key=api_key, base_url=base_url)
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
    if backend == "ollama":
        return _ollama_generate(
            prompt, system,
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            model=kwargs.get("model", "llama3"),
            temperature=temperature,
        )
    elif backend == "openai":
        return _openai_generate(
            prompt, system,
            api_key=kwargs.get("api_key", ""),
            base_url=kwargs.get("base_url", "https://api.openai.com/v1"),
            model=kwargs.get("model", "gpt-4o-mini"),
            temperature=temperature,
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
        fields_to_generate.append("TITLE")
    if gen_lyrics:
        fields_to_generate.append("LYRICS")
    if gen_tags:
        fields_to_generate.append("TAGS")

    if not fields_to_generate:
        return {"title": title, "lyrics": lyrics, "tags": tags}

    # Build context
    context_parts = []
    if description.strip():
        context_parts.append(f"Song description: {description}")
    if title.strip() and not gen_title:
        context_parts.append(f"Song title (already set): {title}")
    if lyrics.strip() and not gen_lyrics:
        context_parts.append(f"Lyrics (already set):\n{lyrics}")
    if tags.strip() and not gen_tags:
        context_parts.append(f"Tags (already set): {tags}")

    # Also provide filled-in fields that ARE being regenerated as "current draft"
    if title.strip() and gen_title:
        context_parts.append(f"Current title draft (to be improved): {title}")
    if lyrics.strip() and gen_lyrics:
        context_parts.append(f"Current lyrics draft (to be improved):\n{lyrics}")
    if tags.strip() and gen_tags:
        context_parts.append(f"Current tags draft (to be improved): {tags}")

    if max_length_sec:
        context_parts.append(
            f"Target song duration: {max_length_sec} seconds. "
            f"Adjust lyrics length accordingly — shorter songs (under 60s) need only 1-2 short sections, "
            f"longer songs (over 120s) can have more verses, a bridge, etc."
        )

    context = "\n\n".join(context_parts)
    fields_str = ", ".join(fields_to_generate)

    prompt = f"""Based on the following context, generate these fields: {fields_str}

{context}

Generate ONLY the requested fields using the exact marker format specified."""

    response = _call_llm(prompt, GENERATE_ALL_SYSTEM_PROMPT, backend, **kwargs)

    result = {"title": title, "lyrics": lyrics, "tags": tags}

    # Parse response
    if gen_title:
        m = re.search(r'===TITLE===\s*(.*?)\s*===END_TITLE===', response, re.DOTALL)
        if m:
            result["title"] = m.group(1).strip()

    if gen_lyrics:
        m = re.search(r'===LYRICS===\s*(.*?)\s*===END_LYRICS===', response, re.DOTALL)
        if m:
            result["lyrics"] = _add_dots_to_abbreviations(m.group(1).strip())

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
