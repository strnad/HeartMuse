import json
import logging
import os
import html as html_mod
import gradio as gr
from config import (
    DEFAULT_GENERATION_PARAMS, DEFAULT_OLLAMA_URL, DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL, DEFAULT_OPENAI_URL, DEFAULT_OPENAI_KEY,
    DEFAULT_OPENAI_MODELS, DEFAULT_LLM_BACKEND, DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT, OUTPUT_DIR,
    DEFAULT_LAZY_LOAD, DEFAULT_MODEL_VARIANT, MODEL_VARIANT_LABELS,
    DEFAULT_NUM_VARIANTS, STYLE_TRANSFER_ENABLED, TRANSCRIPTION_ENABLED,
)
from model_manager import is_ready_for_generation
from lyrics_llm import generate_checked_fields, unload_ollama_model, list_ollama_models
from generator import generate_music, unload_pipeline, cancel_generation, GenerationCancelled, is_generation_cancelled
from history import generate_output_path, save_generation, load_history, delete_generation

logger = logging.getLogger(__name__)

HISTORY_PAGE_SIZE = 10
_JS_SWITCH_TO_GENERATE = "() => document.querySelector('button[role=\"tab\"]').click()"

# Map display labels to internal variant names
_VARIANT_CHOICES = list(MODEL_VARIANT_LABELS.values())
_VARIANT_LABEL_TO_NAME = {v: k for k, v in MODEL_VARIANT_LABELS.items()}


def _variant_name_from_label(label):
    """Convert dropdown label to internal variant name."""
    return _VARIANT_LABEL_TO_NAME.get(label, "rl")


def _variant_label_from_name(name):
    """Convert internal variant name to dropdown label."""
    return MODEL_VARIANT_LABELS.get(name, "HeartMuLa 3B RL (Recommended)")


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
        except Exception as e:
            logger.warning("Failed to unload Ollama model: %s", e)


def _btns_disabled():
    return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True)

def _btns_enabled():
    return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False)


def _status_html(message, style="info"):
    """Return styled HTML for status messages with optional progress bar."""
    message = html_mod.escape(str(message))
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


def _build_variants_html(variants, autoplay=False):
    """Build HTML for one or more audio variant players.

    Args:
        variants: list of (abs_path, filename) tuples
        autoplay: if True, autoplay the last audio element
    """
    if not variants:
        return ""
    if len(variants) == 1:
        path, fname = variants[0]
        autoplay_attr = " autoplay" if autoplay else ""
        return f'<audio controls{autoplay_attr} src="/gradio_api/file={path}" style="width:100%;margin:10px 0;"></audio>'

    parts = []
    for idx, (path, fname) in enumerate(variants, 1):
        is_last = idx == len(variants)
        autoplay_attr = " autoplay" if (autoplay and is_last) else ""
        parts.append(
            f'<div style="margin:8px 0;">'
            f'<span style="color:#a0aec0;font-size:0.9em;font-weight:600;">Variant {idx}</span>'
            f'<span style="color:#666;font-size:0.8em;margin-left:8px;">{html_mod.escape(fname)}</span>'
            f'<audio controls{autoplay_attr} src="/gradio_api/file={path}" style="width:100%;margin:4px 0;"></audio>'
            f'</div>'
        )
    return f'<div>{"".join(parts)}</div>'


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

        result = generate_checked_fields(
            description=description,
            title=title,
            lyrics=lyrics,
            tags=tags,
            gen_desc=gen_desc,
            gen_title=gen_title,
            gen_lyrics=gen_lyrics,
            gen_tags=gen_tags,
            backend=backend.lower(),
            max_length_sec=max_length_sec,
            **kwargs,
        )
        _maybe_unload_ollama(backend, ollama_url, ollama_model, auto_unload)

        # Build status message reporting what actually succeeded vs failed
        requested = [f for f, checked in [("description", gen_desc), ("title", gen_title), ("lyrics", gen_lyrics), ("tags", gen_tags)] if checked]
        failed = result.get("failed_fields", [])
        succeeded = [f for f in requested if f not in failed]

        if failed and succeeded:
            status = _status_html(f"Generated: {', '.join(succeeded)}. Failed to parse: {', '.join(failed)}", "info")
        elif failed and not succeeded:
            status = _status_html(f"LLM did not return expected format for: {', '.join(failed)}", "error")
        else:
            status = _status_html(f"Generated: {', '.join(succeeded)}", "success")

        yield result["description"], result["title"], result["lyrics"], result["tags"], status, *_btns_enabled()
    except Exception as e:
        logger.error("Text generation failed: %s", e)
        yield description, title, lyrics, tags, _status_html(f"Error: {e}", "error"), *_btns_enabled()


def on_generate_music_only(song_title, description, lyrics, tags,
                           temperature, cfg_scale, topk, max_length_sec,
                           lazy_load, model_variant_label, autoplay, num_variants,
                           style_audio, style_enabled, style_strength, seed):
    """Generate music only from current fields (no LLM). Supports batch variants."""
    if not lyrics.strip():
        yield gr.skip(), _status_html("Please enter lyrics.", "error"), *_btns_enabled()
        return
    if not tags.strip() and not (style_enabled and style_audio):
        yield gr.skip(), _status_html("Please enter tags or provide a style reference audio.", "error"), *_btns_enabled()
        return

    num_variants = int(num_variants)
    variant_name = _variant_name_from_label(model_variant_label)
    seed_val = None if seed is None or seed < 0 else int(seed)

    style_embedding = None

    yield gr.skip(), _status_html("Checking models...", "progress"), *_btns_disabled()

    try:
        from generator import ensure_models_downloaded
        from model_manager import is_ready_for_generation
        if not is_ready_for_generation(variant_name):
            yield gr.skip(), _status_html("Downloading required models (this may take a while)...", "progress"), *_btns_disabled()
            ensure_models_downloaded(variant_name)
            if not is_ready_for_generation(variant_name):
                yield gr.skip(), _status_html("Model download failed. Check your internet connection and disk space.", "error"), *_btns_enabled()
                return

        # Extract style embedding from reference audio if enabled
        if style_enabled and style_audio:
            yield gr.skip(), _status_html("Extracting style from reference audio...", "progress"), *_btns_disabled()
            try:
                from style_transfer import extract_style_embedding
                style_embedding = extract_style_embedding(style_audio)
                if style_strength != 1.0:
                    style_embedding = style_embedding * style_strength
            except Exception as e:
                logger.error("Style extraction failed: %s", e)
                yield gr.skip(), _status_html(f"Style extraction failed: {e}", "error"), *_btns_enabled()
                return

        title = song_title.strip() or "Untitled"
        completed = []  # list of (abs_path, filename) tuples

        for i in range(num_variants):
            # Check cancellation between variants
            if i > 0 and is_generation_cancelled():
                yield _build_variants_html(completed, autoplay), \
                      _status_html(f"Cancelled after {len(completed)} of {num_variants} variants.", "info"), \
                      *_btns_enabled()
                return

            if num_variants > 1:
                status_msg = f"Generating variant {i + 1} of {num_variants}..."
            else:
                status_msg = "Generating music (this may take a while)..."
            audio_update = _build_variants_html(completed, autoplay=False) if completed else gr.skip()
            yield audio_update, _status_html(status_msg, "progress"), *_btns_disabled()

            output_path = generate_output_path(title)
            variant_seed = (seed_val + i) if seed_val is not None else None
            path = generate_music(
                lyrics=lyrics,
                tags=tags,
                temperature=temperature,
                cfg_scale=cfg_scale,
                topk=topk,
                max_audio_length_ms=int(max_length_sec * 1000),
                output_path=output_path,
                lazy_load=lazy_load,
                model_variant=variant_name,
                style_embedding=style_embedding,
                seed=variant_seed,
            )

            params = {
                "temperature": temperature,
                "cfg_scale": cfg_scale,
                "topk": topk,
                "max_length_sec": max_length_sec,
                "lazy_load": lazy_load,
                "model_variant": variant_name,
                "style_reference": bool(style_embedding is not None),
                "style_strength": style_strength if style_embedding is not None else None,
                "seed": variant_seed,
            }
            if num_variants > 1:
                params["batch_total"] = num_variants
                params["batch_variant"] = i + 1
            save_generation(
                song_title=title if num_variants == 1 else f"{title} (Variant {i + 1})",
                description=description,
                lyrics=lyrics,
                tags=tags,
                params=params,
                audio_path=path,
            )
            completed.append((path, os.path.basename(path)))

            is_last = i == num_variants - 1
            if is_last:
                if num_variants > 1:
                    msg = f"All {num_variants} variants generated."
                else:
                    msg = f"Music saved as {os.path.basename(path)}"
                yield _build_variants_html(completed, autoplay), \
                      _status_html(msg, "success"), *_btns_enabled()
            else:
                yield _build_variants_html(completed, autoplay=False), \
                      _status_html(f"Variant {i + 1} of {num_variants} complete.", "info"), \
                      *_btns_disabled()

    except GenerationCancelled:
        if completed:
            yield _build_variants_html(completed, autoplay), \
                  _status_html(f"Cancelled after {len(completed)} of {num_variants} variants.", "info"), \
                  *_btns_enabled()
        else:
            yield gr.skip(), _status_html("Generation cancelled.", "info"), *_btns_enabled()
    except Exception as e:
        logger.error("Music generation failed: %s", e)
        e.__traceback__ = None  # release stack frame refs to GPU tensors
        if completed:
            yield _build_variants_html(completed, autoplay), \
                  _status_html(f"Error on variant {len(completed) + 1}: {e}", "error"), \
                  *_btns_enabled()
        else:
            yield gr.skip(), _status_html(f"Error: {e}", "error"), *_btns_enabled()
    finally:
        if style_embedding is not None:
            from style_transfer import unload_muq
            unload_muq()


def on_unload():
    unload_pipeline()
    return _status_html("Pipeline unloaded, GPU memory freed.", "success")


def on_clear_all(gen_desc, gen_title, gen_lyrics, gen_tags, desc, title, lyrics, tags):
    """Clear text fields that have their checkboxes checked."""
    new_desc = "" if gen_desc else desc
    new_title = "" if gen_title else title
    new_lyrics = "" if gen_lyrics else lyrics
    new_tags = "" if gen_tags else tags
    cleared = [name for name, checked in [("description", gen_desc), ("title", gen_title), ("lyrics", gen_lyrics), ("tags", gen_tags)] if checked]
    msg = f"Cleared: {', '.join(cleared)}" if cleared else "Nothing selected to clear."
    return new_desc, new_title, new_lyrics, new_tags, _status_html(msg, "info")


def on_transcribe(audio_path, auto_unload):
    """Transcribe lyrics from uploaded audio file."""
    if not audio_path:
        yield "", _status_html("Please upload an audio file.", "error"), gr.update(interactive=True)
        return

    yield "", _status_html("Checking transcriptor model...", "progress"), gr.update(interactive=False)

    from model_manager import is_transcriptor_downloaded, download_transcriptor
    if not is_transcriptor_downloaded():
        yield "", _status_html("Downloading HeartTranscriptor model (first time only, this may take a while)...", "progress"), gr.update(interactive=False)
        download_transcriptor()
        if not is_transcriptor_downloaded():
            yield "", _status_html("Failed to download transcriptor model. Check internet connection.", "error"), gr.update(interactive=True)
            return

    yield "", _status_html("Transcribing lyrics (this may take a moment)...", "progress"), gr.update(interactive=False)

    try:
        from transcriptor import transcribe_audio
        text = transcribe_audio(audio_path)
        if not text or not text.strip():
            yield "", _status_html("Transcription completed but no lyrics were detected.", "info"), gr.update(interactive=True)
        else:
            yield text, _status_html("Transcription complete.", "success"), gr.update(interactive=True)
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        yield "", _status_html(f"Transcription failed: {e}", "error"), gr.update(interactive=True)
    finally:
        if auto_unload:
            from transcriptor import unload_transcriptor
            unload_transcriptor()


def on_unload_transcriptor():
    from transcriptor import unload_transcriptor
    unload_transcriptor()
    return _status_html("HeartTranscriptor unloaded.", "success")


def _build_card_html(e):
    """Build HTML for a single history card (display only, no interactive elements)."""
    audio_file = e.get("audio_file", "")
    audio_path = os.path.realpath(os.path.join(OUTPUT_DIR, audio_file))
    # Prevent path traversal
    if not audio_path.startswith(os.path.realpath(OUTPUT_DIR) + os.sep):
        audio_path = ""
    title = html_mod.escape(e.get("song_title", "Untitled"))
    ts = e.get("timestamp", "")[:16].replace("T", " ")
    desc = html_mod.escape(e.get("description", ""))
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
        if p.get("style_reference"):
            strength = p.get("style_strength", 1.0)
            strength_label = f"Style: {strength}x" if strength != 1.0 else "Style Ref"
            badges.append(f'<span style="display:inline-block;background:#553c9a;color:#d6bcfa;padding:2px 8px;border-radius:8px;font-size:0.75em;margin:2px;">{strength_label}</span>')
        if p.get("seed") is not None:
            badges.append(f'<span style="display:inline-block;background:#1a365d;color:#63b3ed;padding:2px 8px;border-radius:8px;font-size:0.75em;margin:2px;">Seed: {p["seed"]}</span>')
        for label, key in [("Temp", "temperature"), ("CFG", "cfg_scale"), ("Top-K", "topk"), ("Length", "max_length_sec")]:
            val = p.get(key, "?")
            suffix = "s" if key == "max_length_sec" else ""
            badges.append(f'<span style="display:inline-block;background:#1a365d;color:#63b3ed;padding:2px 8px;border-radius:8px;font-size:0.75em;margin:2px;">{label}: {val}{suffix}</span>')
        param_badges = f'<div style="margin:6px 0;">{" ".join(badges)}</div>'

    return f"""<div style="border:1px solid #444;border-radius:12px;padding:16px;background:rgba(255,255,255,0.02);">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <h3 style="margin:0;font-size:1.1em;">{title}</h3>
    <span style="color:#888;font-size:0.8em;white-space:nowrap;margin-left:12px;">{ts}</span>
  </div>
  {f'<p style="color:#aaa;font-size:0.85em;margin:4px 0;">{desc}</p>' if desc else ''}
  {tag_pills}
  {audio_html}
  {param_badges}
  <details style="margin-top:10px;">
    <summary style="cursor:pointer;color:#7b8cde;font-size:0.85em;">Show full details</summary>
    <div style="margin-top:8px;font-size:0.83em;padding:10px;background:rgba(0,0,0,0.15);border-radius:8px;">
      <p style="margin:4px 0;"><b>Tags:</b> {tags}</p>
      <p style="margin:8px 0 4px;"><b>Lyrics:</b></p>
      <pre style="white-space:pre-wrap;max-height:250px;overflow-y:auto;padding:8px;border-radius:6px;background:rgba(0,0,0,0.25);">{lyrics}</pre>
    </div>
  </details>
</div>"""


def _build_playlist_html(entries):
    """Build HTML-only playlist player. JS is injected via app.launch(js=...)."""
    tracks = []
    real_output = os.path.realpath(OUTPUT_DIR) + os.sep
    for e in entries:
        audio_file = e.get("audio_file", "")
        audio_path = os.path.realpath(os.path.join(OUTPUT_DIR, audio_file))
        if not audio_path.startswith(real_output):
            continue
        if os.path.isfile(audio_path):
            tracks.append({
                "title": e.get("song_title", "Untitled"),
                "ts": e.get("timestamp", "")[:16].replace("T", " "),
                "src": f"/gradio_api/file={audio_path}",
            })
    if not tracks:
        return ""
    tracks_attr = html_mod.escape(json.dumps(tracks))
    return f'''<div id="hm-playlist-player" data-tracks="{tracks_attr}" style="border:1px solid #444;border-radius:12px;padding:16px;background:#1a1a2e;margin-bottom:16px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <div style="display:flex;align-items:center;gap:8px;min-width:0;flex:1;">
      <span style="font-size:1.2em;">&#9835;</span>
      <span id="hm-pl-title" style="color:#e2e8f0;font-weight:600;font-size:1.05em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">&mdash;</span>
    </div>
    <span id="hm-pl-ts" style="color:#888;font-size:0.8em;white-space:nowrap;margin-left:12px;"></span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
    <button id="hm-pl-prev" title="Previous" style="background:#2d3748;color:#e2e8f0;border:1px solid #555;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:0.9em;">&#9664;&#9664;</button>
    <button id="hm-pl-play" title="Play" style="background:#3b82f6;color:#fff;border:none;border-radius:8px;padding:8px 16px;cursor:pointer;font-size:1.1em;min-width:44px;">&#9654;</button>
    <button id="hm-pl-next" title="Next" style="background:#2d3748;color:#e2e8f0;border:1px solid #555;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:0.9em;">&#9654;&#9654;</button>
    <span style="flex:1;"></span>
    <button id="hm-pl-mode" title="Toggle shuffle" style="background:#2d3748;color:#e2e8f0;border:1px solid #555;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:0.8em;">Sequential</button>
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
    <span id="hm-pl-time" style="color:#888;font-size:0.8em;min-width:36px;">0:00</span>
    <div id="hm-pl-bar" style="flex:1;height:6px;background:#2d3748;border-radius:3px;cursor:pointer;position:relative;">
      <div id="hm-pl-fill" style="height:100%;background:linear-gradient(90deg,#3b82f6,#60a5fa);border-radius:3px;width:0%;transition:width 0.1s linear;"></div>
    </div>
    <span id="hm-pl-dur" style="color:#888;font-size:0.8em;min-width:36px;text-align:right;">0:00</span>
  </div>
  <div id="hm-pl-counter" style="text-align:center;color:#666;font-size:0.75em;"></div>
  <audio id="hm-pl-audio" preload="auto"></audio>
</div>'''


# ─── UI ───

CUSTOM_CSS = """
#song-desc-group {
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 12px;
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(59,130,246,0.02));
}
"""

PLAYLIST_JS = """
(function() {
  var SK = '__hmPlState';
  var lastHash = '';
  var audio = null;

  function init() {
    var el = document.getElementById('hm-playlist-player');
    if (!el) return;
    var raw = el.getAttribute('data-tracks');
    if (!raw) return;
    // Skip if tracks haven't changed
    var hash = raw.length + ':' + raw.slice(0, 100);
    if (hash === lastHash && audio && document.contains(audio)) return;
    lastHash = hash;

    var tracks;
    try { tracks = JSON.parse(raw); } catch(e) { return; }
    if (!tracks.length) return;

    var prev = window[SK] || {};
    var ci = (typeof prev.ci === 'number' && prev.ci < tracks.length) ? prev.ci : 0;
    var shuf = prev.shuf || false;
    var shufOrd = prev.shufOrd || [];
    var wasPlay = prev.wasPlay || false;
    var savedT = prev.savedT || 0;
    var savedSrc = prev.savedSrc || '';

    // Create persistent audio element (survives Gradio re-renders)
    if (!audio) {
      audio = document.createElement('audio');
      audio.preload = 'auto';
      audio.style.display = 'none';
      document.body.appendChild(audio);
    }

    var playBtn = document.getElementById('hm-pl-play');
    var prevBtn = document.getElementById('hm-pl-prev');
    var nextBtn = document.getElementById('hm-pl-next');
    var modeBtn = document.getElementById('hm-pl-mode');
    var titleEl = document.getElementById('hm-pl-title');
    var tsEl = document.getElementById('hm-pl-ts');
    var fill = document.getElementById('hm-pl-fill');
    var timeEl = document.getElementById('hm-pl-time');
    var durEl = document.getElementById('hm-pl-dur');
    var ctrEl = document.getElementById('hm-pl-counter');
    var bar = document.getElementById('hm-pl-bar');

    if (!playBtn || !bar) return;

    function fmt(s) {
      if (isNaN(s)) return '0:00';
      var m = Math.floor(s/60), sec = Math.floor(s%60);
      return m + ':' + (sec < 10 ? '0' : '') + sec;
    }
    function genShuf() {
      shufOrd = [];
      for (var i = 0; i < tracks.length; i++) shufOrd.push(i);
      for (var i = shufOrd.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var t = shufOrd[i]; shufOrd[i] = shufOrd[j]; shufOrd[j] = t;
      }
    }
    function ri(idx) { return shuf ? (shufOrd[idx] || 0) : idx; }
    function loadTrack(idx) {
      ci = idx;
      var t = tracks[ri(idx)];
      audio.src = t.src;
      titleEl.textContent = t.title;
      tsEl.textContent = t.ts;
      ctrEl.textContent = 'Track ' + (idx + 1) + ' of ' + tracks.length;
      fill.style.width = '0%';
      timeEl.textContent = '0:00';
      durEl.textContent = '0:00';
    }
    function playTrack(idx) {
      loadTrack(idx);
      audio.play().catch(function(){});
      playBtn.innerHTML = '\\u23F8';
    }
    function save() {
      window[SK] = {
        ci: ci, shuf: shuf, shufOrd: shufOrd,
        wasPlay: !audio.paused, savedT: audio.currentTime || 0, savedSrc: audio.src || ''
      };
    }

    // Sync play button with audio state
    function syncBtn() {
      playBtn.innerHTML = audio.paused ? '\\u25B6' : '\\u23F8';
    }

    playBtn.onclick = function() {
      if (audio.paused) {
        if (!audio.src) loadTrack(0);
        audio.play().catch(function(){});
      } else {
        audio.pause();
      }
      syncBtn();
    };
    nextBtn.onclick = function() { playTrack((ci + 1) % tracks.length); };
    prevBtn.onclick = function() {
      if (audio.currentTime > 3) audio.currentTime = 0;
      else playTrack((ci - 1 + tracks.length) % tracks.length);
    };
    modeBtn.onclick = function() {
      shuf = !shuf;
      modeBtn.textContent = shuf ? 'Shuffle' : 'Sequential';
      modeBtn.style.background = shuf ? '#3b82f6' : '#2d3748';
      if (shuf) genShuf();
      save();
    };

    // Remove old listeners to avoid duplicates
    audio.onended = function() {
      var nx = ci + 1;
      if (nx < tracks.length) playTrack(nx);
      else { if (shuf) genShuf(); playTrack(0); }
    };
    audio.ontimeupdate = function() {
      var f = document.getElementById('hm-pl-fill');
      var te = document.getElementById('hm-pl-time');
      var de = document.getElementById('hm-pl-dur');
      if (f && audio.duration) {
        f.style.width = (audio.currentTime / audio.duration * 100) + '%';
        te.textContent = fmt(audio.currentTime);
        de.textContent = fmt(audio.duration);
      }
      save();
    };
    bar.onclick = function(e) {
      if (audio.duration) {
        var r = bar.getBoundingClientRect();
        audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration;
      }
    };

    // Mutual exclusion with card players
    audio.onplay = function() {
      document.querySelectorAll('audio[controls]').forEach(function(x) {
        if (!x.paused) x.pause();
      });
    };
    if (!window.__hmPlCapture) {
      window.__hmPlCapture = true;
      document.addEventListener('play', function(e) {
        if (audio && e.target !== audio && e.target.tagName === 'AUDIO') {
          audio.pause();
          var pb = document.getElementById('hm-pl-play');
          if (pb) pb.innerHTML = '\\u25B6';
          save();
        }
      }, true);
    }

    // Restore state
    if (shuf) {
      modeBtn.textContent = 'Shuffle';
      modeBtn.style.background = '#3b82f6';
      if (!shufOrd.length || shufOrd.length !== tracks.length) genShuf();
    }

    // If audio is already playing (persistent element), just sync UI
    if (!audio.paused && audio.src) {
      // Find which track matches current src
      for (var i = 0; i < tracks.length; i++) {
        if (audio.src.indexOf(tracks[i].src) !== -1) {
          ci = shuf ? shufOrd.indexOf(i) : i;
          if (ci < 0) ci = 0;
          break;
        }
      }
      var t = tracks[ri(ci)];
      titleEl.textContent = t.title;
      tsEl.textContent = t.ts;
      ctrEl.textContent = 'Track ' + (ci + 1) + ' of ' + tracks.length;
      syncBtn();
    } else {
      loadTrack(ci);
      if (wasPlay && savedSrc) {
        var expected = tracks[ri(ci)] ? tracks[ri(ci)].src : '';
        if (expected && savedSrc.indexOf(expected) !== -1) {
          audio.currentTime = savedT;
          audio.play().catch(function(){});
          syncBtn();
        }
      }
    }
  }

  // Run init now and observe DOM for Gradio updates
  init();
  new MutationObserver(function() { setTimeout(init, 50); })
    .observe(document.body, { childList: true, subtree: true });
})();
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
                    max_lines=10,
                )
                gen_desc_cb = gr.Checkbox(value=False, label="Auto-generate", scale=1)

        # 2. Song Details with checkboxes
        gr.Markdown("### Song details")

        with gr.Row():
            song_title_box = gr.Textbox(label="Song Title", placeholder="Enter or leave empty to generate...", scale=4)
            gen_title_cb = gr.Checkbox(value=True, label="Auto-generate", scale=1)

        with gr.Row():
            lyrics_box = gr.Textbox(
                label="Lyrics",
                placeholder="[verse]\nYour lyrics here...\n\n[chorus]\n...",
                lines=10,
                max_lines=10,
                scale=4,
            )
            gen_lyrics_cb = gr.Checkbox(value=True, label="Auto-generate", scale=1)

        with gr.Row():
            tags_box = gr.Textbox(
                label="Tags (comma-separated)",
                placeholder="pop,female,piano,dreamy,80s,lo-fi,chill,slow",
                scale=4,
            )
            gen_tags_cb = gr.Checkbox(value=True, label="Auto-generate", scale=1)

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

        # Style Reference
        if STYLE_TRANSFER_ENABLED:
            with gr.Accordion("Style Reference (experimental)", open=False):
                gr.Markdown("Upload a reference audio to guide the musical style. "
                            "The model will extract style characteristics and use them during generation.")
                style_audio = gr.Audio(
                    label="Reference Audio",
                    type="filepath",
                    sources=["upload"],
                )
                style_enabled_cb = gr.Checkbox(
                    label="Use style reference",
                    value=False,
                    info="When enabled, the uploaded audio will guide the style of generated music.",
                )
                style_strength_sl = gr.Slider(
                    minimum=0.0, maximum=10.0, value=1.0, step=0.1,
                    label="Style Strength",
                    info="0 = no effect, 1 = normal, >1 = amplified style influence",
                )
        else:
            style_audio = gr.State(value=None)
            style_enabled_cb = gr.State(value=False)
            style_strength_sl = gr.State(value=1.0)

        # Music Generation Settings
        with gr.Accordion("Music Generation Settings", open=False):
            temperature = gr.Slider(0.1, 2.0, value=DEFAULT_GENERATION_PARAMS["temperature"], label="Temperature")
            cfg_scale = gr.Slider(0.0, 5.0, value=DEFAULT_GENERATION_PARAMS["cfg_scale"], label="CFG Scale")
            topk = gr.Slider(1, 200, value=DEFAULT_GENERATION_PARAMS["topk"], step=1, label="Top-K")
            max_length = gr.Slider(10, 240, value=DEFAULT_GENERATION_PARAMS["max_audio_length_ms"] // 1000, step=10, label="Max Length (seconds)")
            seed_box = gr.Number(
                label="Seed",
                value=-1,
                precision=0,
                info="Set to -1 for random. Use the same seed to reproduce identical results.",
            )

        with gr.Row():
            num_variants = gr.Slider(
                1, 10, value=DEFAULT_NUM_VARIANTS, step=1,
                label="Number of Variants",
                info="Generate multiple audio variations from the same lyrics/tags.",
            )
            autoplay_cb = gr.Checkbox(label="Autoplay", value=True)
        with gr.Row():
            gen_music_btn = gr.Button("Generate Music", variant="primary", size="lg")
            cancel_btn = gr.Button("Cancel", variant="stop", size="lg", interactive=False)

        # Output - styled HTML status
        status_box = gr.HTML(value="", label="Status")
        audio_out = gr.HTML(value="")

        # Memory Management
        with gr.Accordion("Memory Management", open=False):
            model_variant = gr.Dropdown(
                choices=_VARIANT_CHOICES,
                value=_variant_label_from_name(DEFAULT_MODEL_VARIANT),
                label="Model",
                info="Select which HeartMuLa model to use. RL version has better style/tag control.",
            )
            lazy_load_cb = gr.Checkbox(
                value=DEFAULT_LAZY_LOAD,
                label="Lazy Loading",
                info="Load models on demand and free VRAM between generation stages. Useful for GPUs with limited memory.",
            )
            ollama_auto_unload = gr.Checkbox(value=True, label="Auto-unload Ollama model before music generation")
            with gr.Row():
                unload_ollama_btn = gr.Button("Unload Ollama Model", size="sm")
                unload_btn = gr.Button("Unload Music Pipeline", size="sm")
                unload_muq_btn = gr.Button("Unload MuQ Model", size="sm")
                unload_transcriptor_gen_btn = gr.Button("Unload Transcriptor", size="sm")
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
        music_inputs = [temperature, cfg_scale, topk, max_length, lazy_load_cb, model_variant, autoplay_cb, num_variants, style_audio, style_enabled_cb, style_strength_sl, seed_box]

        all_btns = [gen_text_btn, gen_music_btn, cancel_btn]

        # Generate Text button
        gen_text_btn.click(
            on_generate_text,
            [song_desc, song_title_box, lyrics_box, tags_box,
             gen_desc_cb, gen_title_cb, gen_lyrics_cb, gen_tags_cb] + llm_inputs + [max_length],
            [song_desc, song_title_box, lyrics_box, tags_box, status_box] + all_btns,
        )

        # Generate Music button
        gen_music_btn.click(
            on_generate_music_only,
            [song_title_box, song_desc, lyrics_box, tags_box] + music_inputs,
            [audio_out, status_box] + all_btns,
        )

        unload_ollama_btn.click(on_unload_ollama, [llm_backend, ollama_url, ollama_model], [ollama_status])
        unload_btn.click(on_unload, [], [status_box])
        def on_unload_muq():
            from style_transfer import unload_muq
            unload_muq()
            return _status_html("MuQ model unloaded.", "success")
        unload_muq_btn.click(on_unload_muq, [], [status_box])
        unload_transcriptor_gen_btn.click(on_unload_transcriptor, [], [status_box])
        cancel_btn.click(lambda: cancel_generation(), [], [])

        # Clear button
        clear_btn.click(on_clear_all, [gen_desc_cb, gen_title_cb, gen_lyrics_cb, gen_tags_cb, song_desc, song_title_box, lyrics_box, tags_box], [song_desc, song_title_box, lyrics_box, tags_box, status_box])

    if TRANSCRIPTION_ENABLED:
        with gr.Tab("Transcribe"):
            gr.Markdown("### Transcribe Lyrics from Audio")
            gr.Markdown(
                "Upload an audio file to transcribe its lyrics using HeartTranscriptor. "
                "For best results, use source-separated vocal tracks "
                "(e.g., processed with [demucs](https://github.com/adefossez/demucs))."
            )

            transcribe_audio_input = gr.Audio(
                label="Audio to Transcribe",
                type="filepath",
                sources=["upload"],
            )

            transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")
            transcribe_auto_unload = gr.Checkbox(
                value=True,
                label="Auto-unload model after transcription",
                info="Free GPU/RAM memory automatically when transcription finishes.",
            )
            transcribe_status = gr.HTML(value="")

            transcribed_lyrics = gr.Textbox(
                label="Transcribed Lyrics",
                lines=12,
                max_lines=20,
                interactive=True,
                placeholder="Transcribed lyrics will appear here...",
            )

            send_to_generator_btn = gr.Button("Send to Generator", variant="secondary", size="sm")

            with gr.Accordion("Memory Management", open=False):
                gr.Markdown("HeartTranscriptor uses GPU VRAM (or RAM on CPU). "
                            "Unload it before generating music if GPU memory is limited.")
                unload_transcriptor_btn = gr.Button("Unload Transcriptor", size="sm")

            transcribe_btn.click(
                on_transcribe,
                inputs=[transcribe_audio_input, transcribe_auto_unload],
                outputs=[transcribed_lyrics, transcribe_status, transcribe_btn],
            )

            unload_transcriptor_btn.click(on_unload_transcriptor, [], [transcribe_status])

            def _send_lyrics_to_generator(transcribed_text):
                if not transcribed_text or not transcribed_text.strip():
                    return gr.skip(), _status_html("Nothing to send.", "info")
                return transcribed_text, _status_html("Lyrics sent to Generator tab.", "success")

            send_to_generator_btn.click(
                _send_lyrics_to_generator,
                inputs=[transcribed_lyrics],
                outputs=[lyrics_box, transcribe_status],
            ).then(fn=None, js=_JS_SWITCH_TO_GENERATE)

    with gr.Tab("History") as history_tab:
        history_page = gr.State(value=0)
        playlist_player = gr.HTML(value="")
        history_status = gr.HTML(value="")
        page_info = gr.HTML(value="")

        # Fixed card slots (avoids @gr.render and its stale-handler bugs)
        _card_htmls = []
        _card_states = []
        _load_btns = []
        _delete_btns = []
        for _i in range(HISTORY_PAGE_SIZE):
            _st = gr.State(value=None)
            _card_states.append(_st)
            _html = gr.HTML(visible=False)
            with gr.Row(visible=False) as _row:
                _lb = gr.Button("Load to Generator", size="sm", variant="primary", scale=1)
                _db = gr.Button("Delete", size="sm", variant="stop", scale=1)
            _card_htmls.append((_html, _row))
            _load_btns.append(_lb)
            _delete_btns.append(_db)

        with gr.Row():
            prev_btn = gr.Button("Previous", size="sm")
            next_btn = gr.Button("Next", size="sm")

        # Card-only outputs (for page navigation — no playlist rebuild)
        _card_outputs = []
        for _i in range(HISTORY_PAGE_SIZE):
            _html_comp, _row_comp = _card_htmls[_i]
            _card_outputs.extend([_html_comp, _row_comp, _card_states[_i]])
        _card_outputs.extend([page_info, history_page])

        # Full outputs including playlist player
        _refresh_outputs = [playlist_player] + _card_outputs

        def _build_cards(page, entries):
            """Build card slot updates for a given page."""
            total = len(entries)
            max_page = max(0, (total - 1) // HISTORY_PAGE_SIZE) if total > 0 else 0
            page = min(page, max_page)
            start = page * HISTORY_PAGE_SIZE
            page_entries = entries[start:start + HISTORY_PAGE_SIZE]

            updates = []
            for i in range(HISTORY_PAGE_SIZE):
                if i < len(page_entries):
                    e = page_entries[i]
                    updates.append(gr.HTML(value=_build_card_html(e), visible=True))
                    updates.append(gr.Row(visible=True))
                    updates.append(e)
                else:
                    updates.append(gr.HTML(value="", visible=False))
                    updates.append(gr.Row(visible=False))
                    updates.append(None)

            if total > HISTORY_PAGE_SIZE:
                total_pages = (total + HISTORY_PAGE_SIZE - 1) // HISTORY_PAGE_SIZE
                info = f'<p style="text-align:center;color:#888;font-size:0.85em;">Showing {start+1}-{min(start + HISTORY_PAGE_SIZE, total)} of {total} (page {page+1}/{total_pages})</p>'
            elif total == 0:
                info = '<p style="text-align:center;color:#888;padding:40px 0;">No generations yet.</p>'
            else:
                info = ""
            updates.append(gr.HTML(value=info))
            updates.append(page)
            return updates

        def _refresh_history(page):
            """Full refresh: playlist player + all card slots."""
            page = int(page or 0)
            entries = load_history()
            return [_build_playlist_html(entries)] + _build_cards(page, entries)

        def _refresh_cards_only(page):
            """Cards-only refresh for page navigation (keeps playlist playing)."""
            page = int(page or 0)
            entries = load_history()
            return _build_cards(page, entries)

        # Wire up load handlers (read entry data from per-slot State)
        def _slot_load(state):
            if not state:
                return gr.skip(), gr.skip(), gr.skip(), gr.skip(), _status_html("No entry selected.", "warning")
            return (
                state.get("description", ""),
                state.get("song_title", ""),
                state.get("lyrics", ""),
                state.get("tags", ""),
                _status_html(f"Loaded '{state.get('song_title', 'Untitled')}' into Generator tab.", "success"),
            )

        # Wire up delete handlers (delete files, then refresh page)
        def _slot_delete(state, page):
            if not state:
                return page, _status_html("No entry to delete.", "warning")
            delete_generation(state.get("audio_file", ""))
            return page, _status_html(f"Deleted '{state.get('song_title', 'Untitled')}'.", "success")

        for _i in range(HISTORY_PAGE_SIZE):
            _load_btns[_i].click(
                _slot_load,
                [_card_states[_i]],
                [song_desc, song_title_box, lyrics_box, tags_box, history_status],
            ).then(fn=None, js=_JS_SWITCH_TO_GENERATE)
            _delete_btns[_i].click(
                _slot_delete,
                [_card_states[_i], history_page],
                [history_page, history_status],
                js="(...args) => { if (!confirm('Are you sure you want to delete this song?')) throw new Error('cancelled'); return args; }",
            ).then(
                _refresh_history, [history_page], _refresh_outputs,
            )

        # Navigation
        def _go_prev(page):
            return max(0, int(page or 0) - 1)
        def _go_next(page):
            page = int(page or 0)
            total = len(load_history())
            max_page = max(0, (total - 1) // HISTORY_PAGE_SIZE)
            return min(max_page, page + 1)

        prev_btn.click(_go_prev, [history_page], [history_page]).then(
            _refresh_cards_only, [history_page], _card_outputs)
        next_btn.click(_go_next, [history_page], [history_page]).then(
            _refresh_cards_only, [history_page], _card_outputs)

        # Refresh when switching to this tab
        history_tab.select(_refresh_history, [history_page], _refresh_outputs)


if __name__ == "__main__":
    import sys
    share = "--share" in sys.argv
    from config import SERVER_HOST, SERVER_PORT
    app.launch(share=share, server_name=SERVER_HOST, server_port=SERVER_PORT, allowed_paths=[OUTPUT_DIR], js=PLAYLIST_JS)
