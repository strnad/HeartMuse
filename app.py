import os
import re
import html as html_mod
import gradio as gr
from config import (
    DEFAULT_GENERATION_PARAMS, DEFAULT_OLLAMA_URL, DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL, DEFAULT_OPENAI_URL, DEFAULT_OPENAI_KEY,
    DEFAULT_OPENAI_MODELS, DEFAULT_LLM_BACKEND, DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT, OUTPUT_DIR,
)
from model_manager import is_ready_for_generation
from lyrics_llm import generate_checked_fields, unload_ollama_model, list_ollama_models, _call_llm
from generator import generate_music, unload_pipeline
from history import generate_output_path, save_generation, load_history
from prompt_templates import PromptBuilder


def on_list_ollama(ollama_url):
    models = list_ollama_models(base_url=ollama_url)
    if not models:
        return gr.update(choices=[], value=None)
    return gr.update(choices=models, value=models[0])


def _llm_kwargs(backend, ollama_url, ollama_model, openai_url, openai_model, openai_key, temperature, timeout):
    if backend == "Ollama":
        return {"base_url": ollama_url, "model": ollama_model, "temperature": temperature, "timeout": timeout}
    else:
        return {"api_key": openai_key, "base_url": openai_url, "model": openai_model, "temperature": temperature, "timeout": timeout}


def _maybe_unload_ollama(backend, ollama_url, ollama_model, auto_unload):
    if auto_unload and backend == "Ollama":
        try:
            unload_ollama_model(base_url=ollama_url, model=ollama_model)
        except Exception:
            pass


def _btns_disabled():
    return gr.update(interactive=False), gr.update(interactive=False)

def _btns_enabled():
    return gr.update(interactive=True), gr.update(interactive=True)


def _status_html(message, style="info"):
    """Return styled HTML for status messages with optional progress bar."""
    colors = {
        "info": ("#1a3a5c", "#3b82f6", "#e0f2fe"),
        "success": ("#14532d", "#22c55e", "#dcfce7"),
        "error": ("#7f1d1d", "#ef4444", "#fee2e2"),
        "progress": ("#1a3a5c", "#3b82f6", "#e0f2fe"),
    }
    bg, border, text_bg = colors.get(style, colors["info"])

    progress_bar = ""
    if style == "progress":
        progress_bar = """
        <div style="width:100%;height:4px;background:#1e293b;border-radius:2px;overflow:hidden;margin-top:8px;">
          <div style="width:30%;height:100%;background:linear-gradient(90deg,#3b82f6,#60a5fa);border-radius:2px;animation:progress-slide 1.5s ease-in-out infinite;"></div>
        </div>
        <style>
          @keyframes progress-slide {
            0% { margin-left: 0%; width: 30%; }
            50% { margin-left: 35%; width: 40%; }
            100% { margin-left: 70%; width: 30%; }
          }
        </style>"""

    return f"""<div style="padding:12px 16px;border-left:4px solid {border};background:{bg};border-radius:8px;font-size:1.05em;color:#e2e8f0;">
  {message}{progress_bar}
</div>"""


def on_generate_text(description, title, lyrics, tags,
                     gen_desc, gen_title, gen_lyrics, gen_tags,
                     backend, ollama_url, ollama_model,
                     openai_url, openai_model, openai_key,
                     llm_temp, llm_timeout, auto_unload, max_length_sec):
    """Generate checked text fields using LLM. Can generate from scratch if nothing is provided."""
    if not gen_desc and not gen_title and not gen_lyrics and not gen_tags:
        yield description, title, lyrics, tags, _status_html("Nothing selected for generation.", "error"), *_btns_enabled()
        return

    yield description, title, lyrics, tags, _status_html("Generating text...", "progress"), *_btns_disabled()

    try:
        kwargs = _llm_kwargs(backend, ollama_url, ollama_model, openai_url, openai_model, openai_key, llm_temp, llm_timeout)

        # Generate or enhance description if requested
        final_description = description
        if gen_desc:
            # Build context dict for PromptBuilder
            context = {
                "description": description,
                "title": title,
                "lyrics": lyrics,
                "tags": tags,
            }

            # Use PromptBuilder to create prompts
            system_prompt = PromptBuilder.build_system_prompt(["description"])
            user_prompt = PromptBuilder.build_user_prompt(["description"], context, max_length_sec)

            # Generate description using LLM
            response = _call_llm(user_prompt, system_prompt, backend.lower(), **kwargs)

            # Parse description from response
            m = re.search(r'===DESCRIPTION===\s*(.*?)\s*===END_DESCRIPTION===', response, re.DOTALL)
            if m:
                final_description = m.group(1).strip()
            else:
                # Fallback: use entire response if markers not found
                final_description = response.strip()

        result = generate_checked_fields(
            description=final_description,
            title=title,
            lyrics=lyrics,
            tags=tags,
            gen_title=gen_title,
            gen_lyrics=gen_lyrics,
            gen_tags=gen_tags,
            backend=backend.lower(),
            max_length_sec=max_length_sec,
            **kwargs,
        )
        _maybe_unload_ollama(backend, ollama_url, ollama_model, auto_unload)
        generated = [f for f, checked in [("description", gen_desc), ("title", gen_title), ("lyrics", gen_lyrics), ("tags", gen_tags)] if checked]
        yield final_description, result["title"], result["lyrics"], result["tags"], _status_html(f"Generated: {', '.join(generated)}", "success"), *_btns_enabled()
    except Exception as e:
        yield description, title, lyrics, tags, _status_html(f"Error: {e}", "error"), *_btns_enabled()


def on_generate_music_only(song_title, description, lyrics, tags,
                           temperature, cfg_scale, topk, max_length_sec, lazy_load):
    """Generate music only from current fields (no LLM)."""
    if not lyrics.strip():
        yield None, _status_html("Please enter lyrics.", "error"), *_btns_enabled()
        return
    if not tags.strip():
        yield None, _status_html("Please enter tags.", "error"), *_btns_enabled()
        return

    yield None, _status_html("Checking models...", "progress"), *_btns_disabled()

    try:
        from generator import ensure_models_downloaded
        from model_manager import is_ready_for_generation
        if not is_ready_for_generation():
            yield None, _status_html("Downloading required models (this may take a while)...", "progress"), *_btns_disabled()
            ensure_models_downloaded()

        yield None, _status_html("Generating music (this may take a while)...", "progress"), *_btns_disabled()

        title = song_title.strip() or "Untitled"
        output_path = generate_output_path(title)
        path = generate_music(
            lyrics=lyrics,
            tags=tags,
            temperature=temperature,
            cfg_scale=cfg_scale,
            topk=topk,
            max_audio_length_ms=int(max_length_sec * 1000),
            lazy_load=lazy_load,
            output_path=output_path,
        )
        save_generation(
            song_title=title,
            description=description,
            lyrics=lyrics,
            tags=tags,
            params={
                "temperature": temperature,
                "cfg_scale": cfg_scale,
                "topk": topk,
                "max_length_sec": max_length_sec,
                "lazy_load": lazy_load,
            },
            audio_path=path,
        )
        yield path, _status_html(f"Music saved as {os.path.basename(path)}", "success"), *_btns_enabled()
    except Exception as e:
        yield None, _status_html(f"Error: {e}", "error"), *_btns_enabled()


def on_unload():
    unload_pipeline()
    return _status_html("Pipeline unloaded, GPU memory freed.", "success")


def on_clear_all():
    """Clear all text fields."""
    return "", "", "", "", _status_html("All fields cleared.", "info")


def render_history():
    import json as json_mod
    entries = load_history()
    if not entries:
        return "<p style='text-align:center;color:#888;padding:40px 0;'>No generations yet.</p>"

    cards = []
    for i, e in enumerate(entries):
        audio_file = e.get("audio_file", "")
        audio_path = os.path.abspath(os.path.join(OUTPUT_DIR, audio_file))
        title = html_mod.escape(e.get("song_title", "Untitled"))
        ts = e.get("timestamp", "")[:16].replace("T", " ")
        desc = html_mod.escape(e.get("description", ""))
        desc_short = html_mod.escape(e.get("description", "")[:120])
        tags = html_mod.escape(e.get("tags", ""))
        lyrics = html_mod.escape(e.get("lyrics", ""))
        p = e.get("parameters", {})

        audio_html = ""
        if os.path.isfile(audio_path):
            audio_html = f'<audio controls src="/gradio_api/file={audio_path}" style="width:100%;margin:10px 0;"></audio>'

        tag_pills = ""
        if e.get("tags", ""):
            pills = [f'<span style="display:inline-block;background:#2d3748;color:#a0aec0;padding:2px 8px;border-radius:12px;font-size:0.75em;margin:2px;">{html_mod.escape(t.strip())}</span>'
                     for t in e.get("tags", "").split(",")[:8] if t.strip()]
            tag_pills = f'<div style="margin:6px 0;">{" ".join(pills)}</div>'

        param_badges = ""
        if p:
            badges = []
            for label, key in [("Temp", "temperature"), ("CFG", "cfg_scale"), ("Top-K", "topk"), ("Length", "max_length_sec")]:
                val = p.get(key, "?")
                suffix = "s" if key == "max_length_sec" else ""
                badges.append(f'<span style="display:inline-block;background:#1a365d;color:#63b3ed;padding:2px 8px;border-radius:8px;font-size:0.75em;margin:2px;">{label}: {val}{suffix}</span>')
            param_badges = f'<div style="margin:6px 0;">{" ".join(badges)}</div>'

        cards.append(f"""
<div style="border:1px solid #444;border-radius:12px;padding:16px;margin:10px 0;background:rgba(255,255,255,0.02);transition:border-color 0.2s;" onmouseover="this.style.borderColor='#666'" onmouseout="this.style.borderColor='#444'">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <h3 style="margin:0;font-size:1.1em;">{title}</h3>
    <span style="color:#888;font-size:0.8em;white-space:nowrap;margin-left:12px;">{ts}</span>
  </div>
  {f'<p style="color:#aaa;font-size:0.85em;margin:4px 0;">{desc_short}</p>' if desc_short else ''}
  {tag_pills}
  {audio_html}
  {param_badges}
  <details style="margin-top:10px;">
    <summary style="cursor:pointer;color:#7b8cde;font-size:0.85em;">Show full details</summary>
    <div style="margin-top:8px;font-size:0.83em;padding:10px;background:rgba(0,0,0,0.15);border-radius:8px;">
      {f'<p style="margin:4px 0;"><b>Description:</b> {desc}</p>' if desc else ''}
      <p style="margin:4px 0;"><b>Tags:</b> {tags}</p>
      <p style="margin:8px 0 4px;"><b>Lyrics:</b></p>
      <pre style="white-space:pre-wrap;max-height:250px;overflow-y:auto;padding:8px;border-radius:6px;background:rgba(0,0,0,0.25);">{lyrics}</pre>
    </div>
  </details>
</div>""")
    return "\n".join(cards)


# ─── UI ───

CUSTOM_CSS = """
#song-desc-group {
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 12px;
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(59,130,246,0.02));
}
"""

with gr.Blocks(title="HeartMuse Music Generator", css=CUSTOM_CSS) as app:
    gr.Markdown("# HeartMuse Music Generator")

    with gr.Tab("Generate"):
        # 1. Song Description (highlighted)
        gr.Markdown("### Describe your song")
        with gr.Group(elem_id="song-desc-group"):
            with gr.Row():
                song_desc = gr.Textbox(
                    label="Song Description",
                    placeholder="A romantic ballad about summer love with piano and strings... (or leave empty to generate randomly)",
                    lines=3,
                    scale=4,
                )
                gen_desc_cb = gr.Checkbox(value=False, label="Generate/Enhance", scale=1)

        # 2. Song Details with checkboxes
        gr.Markdown("### Song details")

        with gr.Row():
            song_title_box = gr.Textbox(label="Song Title", placeholder="Enter or leave empty to generate...", scale=4)
            gen_title_cb = gr.Checkbox(value=True, label="Generate/Enhance", scale=1)

        with gr.Row():
            lyrics_box = gr.Textbox(
                label="Lyrics",
                placeholder="[verse]\nYour lyrics here...\n\n[chorus]\n...",
                lines=10,
                scale=4,
            )
            gen_lyrics_cb = gr.Checkbox(value=True, label="Generate/Enhance", scale=1)

        with gr.Row():
            tags_box = gr.Textbox(
                label="Tags (comma-separated)",
                placeholder="pop,piano,happy,female vocal",
                scale=4,
            )
            gen_tags_cb = gr.Checkbox(value=True, label="Generate/Enhance", scale=1)

        # LLM Settings (moved above Generate Text button)
        with gr.Accordion("LLM Settings", open=False):
            llm_backend = gr.Radio(["Ollama", "OpenAI"], value=DEFAULT_LLM_BACKEND, label="Backend")
            with gr.Group(visible=DEFAULT_LLM_BACKEND == "Ollama") as ollama_group:
                ollama_url = gr.Textbox(label="Ollama URL", value=DEFAULT_OLLAMA_URL)
                with gr.Row():
                    ollama_model = gr.Dropdown(
                        label="Ollama Model", value=DEFAULT_OLLAMA_MODEL,
                        choices=[DEFAULT_OLLAMA_MODEL], allow_custom_value=True,
                    )
                    refresh_ollama_btn = gr.Button("Refresh Models", size="sm")
                refresh_ollama_btn.click(on_list_ollama, [ollama_url], [ollama_model])
            with gr.Group(visible=DEFAULT_LLM_BACKEND == "OpenAI") as openai_group:
                openai_url = gr.Textbox(label="API Base URL", value=DEFAULT_OPENAI_URL)
                openai_model = gr.Dropdown(
                    label="Model", value=DEFAULT_OPENAI_MODEL,
                    choices=DEFAULT_OPENAI_MODELS, allow_custom_value=True,
                )
                openai_key = gr.Textbox(label="API Key", value=DEFAULT_OPENAI_KEY, type="password")

            def toggle_backend(choice):
                return (
                    gr.update(visible=choice == "Ollama"),
                    gr.update(visible=choice == "OpenAI"),
                )

            llm_backend.change(toggle_backend, [llm_backend], [ollama_group, openai_group])
            llm_temp = gr.Slider(0.0, 2.0, value=DEFAULT_LLM_TEMPERATURE, step=0.1, label="LLM Temperature")
            llm_timeout = gr.Slider(10, 600, value=DEFAULT_LLM_TIMEOUT, step=10, label="LLM Timeout (seconds)")

        # Clear and Generate text buttons
        clear_btn = gr.Button("Clear All", size="sm", variant="secondary")
        gen_text_btn = gr.Button("Generate Text", variant="primary", size="lg")

        # 3. Music generation
        gr.Markdown("### Generate music")

        # Music Generation Settings
        with gr.Accordion("Music Generation Settings", open=False):
            temperature = gr.Slider(0.1, 2.0, value=DEFAULT_GENERATION_PARAMS["temperature"], label="Temperature")
            cfg_scale = gr.Slider(0.0, 5.0, value=DEFAULT_GENERATION_PARAMS["cfg_scale"], label="CFG Scale")
            topk = gr.Slider(1, 200, value=DEFAULT_GENERATION_PARAMS["topk"], step=1, label="Top-K")
            max_length = gr.Slider(10, 240, value=DEFAULT_GENERATION_PARAMS["max_audio_length_ms"] // 1000, step=10, label="Max Length (seconds)")

        gen_music_btn = gr.Button("Generate Music", variant="primary", size="lg")

        # Output - styled HTML status
        status_box = gr.HTML(value="", label="Status")
        audio_out = gr.Audio(label="Generated Music", type="filepath")

        # Memory Management
        with gr.Accordion("Memory Management", open=False):
            lazy_load = gr.Checkbox(value=True, label="Lazy Load (lower VRAM usage)")
            ollama_auto_unload = gr.Checkbox(value=True, label="Auto-unload Ollama model before music generation")
            with gr.Row():
                unload_ollama_btn = gr.Button("Unload Ollama Model", size="sm")
                unload_btn = gr.Button("Unload Music Pipeline", size="sm")
            ollama_status = gr.Textbox(label="Status", interactive=False, visible=False)

        def on_unload_ollama(backend, o_url, o_model):
            if backend != "Ollama":
                return gr.update(visible=True, value="Only applicable for Ollama backend.")
            result = unload_ollama_model(base_url=o_url, model=o_model)
            return gr.update(visible=True, value=result)

        # Common inputs
        llm_inputs = [
            llm_backend, ollama_url, ollama_model,
            openai_url, openai_model, openai_key,
            llm_temp, llm_timeout, ollama_auto_unload,
        ]
        music_inputs = [temperature, cfg_scale, topk, max_length, lazy_load]

        all_btns = [gen_text_btn, gen_music_btn]

        # Generate Text button
        gen_text_event = gen_text_btn.click(
            on_generate_text,
            [song_desc, song_title_box, lyrics_box, tags_box,
             gen_desc_cb, gen_title_cb, gen_lyrics_cb, gen_tags_cb] + llm_inputs + [max_length],
            [song_desc, song_title_box, lyrics_box, tags_box, status_box] + all_btns,
        )

        # Generate Music button
        gen_music_event = gen_music_btn.click(
            on_generate_music_only,
            [song_title_box, song_desc, lyrics_box, tags_box] + music_inputs,
            [audio_out, status_box] + all_btns,
        )

        unload_ollama_btn.click(on_unload_ollama, [llm_backend, ollama_url, ollama_model], [ollama_status])
        unload_btn.click(on_unload, [], [status_box])

        # Clear button
        clear_btn.click(on_clear_all, [], [song_desc, song_title_box, lyrics_box, tags_box, status_box])

    with gr.Tab("History") as history_tab:
        history_html = gr.HTML(value=render_history())

        # Auto-refresh when switching to History tab
        history_tab.select(render_history, [], [history_html])


if __name__ == "__main__":
    import sys
    share = "--share" in sys.argv
    from config import SERVER_HOST, SERVER_PORT
    app.launch(share=share, server_name=SERVER_HOST, server_port=SERVER_PORT, allowed_paths=[OUTPUT_DIR])
