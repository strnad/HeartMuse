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
    DEFAULT_AUDIOSR_ENABLED, DEFAULT_AUDIOSR_DDIM_STEPS,
    DEFAULT_AUDIOSR_GUIDANCE_SCALE, DEFAULT_AUDIOSR_SEED,
    DEFAULT_AUDIOSR_FORMAT, AUDIOSR_FORMAT_CHOICES, AUDIOSR_FORMAT_MAP,
    AUDIOSR_FORMAT_LABELS,
)
from model_manager import is_ready_for_generation
from lyrics_llm import generate_checked_fields, unload_ollama_model, list_ollama_models
from generator import generate_music, unload_pipeline, cancel_generation, GenerationCancelled
from upscaler import upscale_audio, unload_audiosr, cancel_upscale, UpscaleCancelled
from history import (
    generate_output_path, save_generation, load_history, delete_generation,
    update_generation, get_upscaled_files, next_upscale_path,
)

logger = logging.getLogger(__name__)

HISTORY_PAGE_SIZE = 10

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
                           lazy_load, model_variant_label, autoplay,
                           upscale_enabled, upscale_format_label,
                           upscale_ddim_steps, upscale_guidance, upscale_seed):
    """Generate music only from current fields (no LLM)."""
    if not lyrics.strip():
        yield gr.skip(), _status_html("Please enter lyrics.", "error"), *_btns_enabled()
        return
    if not tags.strip():
        yield gr.skip(), _status_html("Please enter tags.", "error"), *_btns_enabled()
        return

    variant_name = _variant_name_from_label(model_variant_label)

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

        yield gr.skip(), _status_html("Generating music (this may take a while)...", "progress"), *_btns_disabled()

        title = song_title.strip() or "Untitled"
        output_path = generate_output_path(title)
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
                "model_variant": variant_name,
            },
            audio_path=path,
        )
        # Free HeartMuLa VRAM before upscaling when lazy_load is enabled
        if lazy_load and upscale_enabled:
            unload_pipeline()

        # Auto-upscale if enabled
        upscaled_path = None
        upscale_error = None
        if upscale_enabled:
            yield gr.skip(), _status_html("Upscaling audio to 48kHz (this may take a while)...", "progress"), *_btns_disabled()
            try:
                fmt = AUDIOSR_FORMAT_MAP.get(upscale_format_label, "flac")
                upscaled_stem = os.path.splitext(path)[0]
                upscaled_path = f"{upscaled_stem}_48kHz.{fmt}"
                upscale_audio(
                    input_path=path,
                    output_path=upscaled_path,
                    seed=int(upscale_seed),
                    ddim_steps=int(upscale_ddim_steps),
                    guidance_scale=upscale_guidance,
                    output_format=fmt,
                )
                update_generation(os.path.basename(path), {
                    "upscaled_files": [{
                        "file": os.path.basename(upscaled_path),
                        "ddim_steps": int(upscale_ddim_steps),
                        "guidance_scale": upscale_guidance,
                        "seed": int(upscale_seed),
                        "format": fmt,
                    }],
                })
                if lazy_load:
                    unload_audiosr()
            except UpscaleCancelled:
                yield gr.skip(), _status_html("Upscaling cancelled. Original audio saved.", "info"), *_btns_enabled()
                return
            except Exception as e:
                logger.error("Upscaling failed: %s", e)
                upscaled_path = None
                upscale_error = str(e)

        # Build audio output
        autoplay_attr = " autoplay" if autoplay else ""
        if upscaled_path and os.path.isfile(upscaled_path):
            audio_html = f'<audio controls{autoplay_attr} src="/gradio_api/file={upscaled_path}" style="width:100%;margin:10px 0;"></audio>'
            audio_html += f'<div style="font-size:0.8em;color:#888;margin-top:4px;">48kHz upscaled version. <a href="/gradio_api/file={path}" download style="color:#63b3ed;">Download original</a></div>'
            status_msg = f"Music generated and upscaled to 48kHz."
        else:
            audio_html = f'<audio controls{autoplay_attr} src="/gradio_api/file={path}" style="width:100%;margin:10px 0;"></audio>'
            status_msg = f"Music saved as {os.path.basename(path)}"
            if upscale_enabled and upscaled_path is None:
                status_msg += f" (upscaling failed: {upscale_error})" if upscale_error else " (upscaling failed)"

        yield audio_html, _status_html(status_msg, "success"), *_btns_enabled()
    except GenerationCancelled:
        yield gr.skip(), _status_html("Generation cancelled.", "info"), *_btns_enabled()
    except Exception as e:
        logger.error("Music generation failed: %s", e)
        e.__traceback__ = None  # release stack frame refs to GPU tensors
        yield gr.skip(), _status_html(f"Error: {e}", "error"), *_btns_enabled()


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


def _build_card_html(e):
    """Build HTML for a single history card (display only, no interactive elements)."""
    audio_file = e.get("audio_file", "")
    audio_path = os.path.abspath(os.path.join(OUTPUT_DIR, audio_file))
    title = html_mod.escape(e.get("song_title", "Untitled"))
    ts = e.get("timestamp", "")[:16].replace("T", " ")
    desc = html_mod.escape(e.get("description", ""))
    tags = html_mod.escape(e.get("tags", ""))
    lyrics = html_mod.escape(e.get("lyrics", ""))
    p = e.get("parameters", {})

    audio_html = ""
    if os.path.isfile(audio_path):
        audio_html = f'<audio controls src="/gradio_api/file={audio_path}" style="width:100%;margin:10px 0;"></audio>'

    upscale_section = ""
    for ui in get_upscaled_files(e):
        fname = ui.get("file", "") if isinstance(ui, dict) else ui
        if not fname:
            continue
        up_path = os.path.abspath(os.path.join(OUTPUT_DIR, fname))
        if os.path.isfile(up_path):
            badge_parts = ["48kHz"]
            if isinstance(ui, dict):
                if ui.get("format"):
                    badge_parts.append(ui["format"].upper())
                if ui.get("ddim_steps"):
                    badge_parts.append(f"{ui['ddim_steps']} steps")
            badge = html_mod.escape(" \u00b7 ".join(badge_parts))
            upscale_section += f'''<div style="margin:6px 0;">
  <span style="display:inline-block;background:#14532d;color:#22c55e;padding:2px 8px;border-radius:8px;font-size:0.75em;">{badge}</span>
  <audio controls src="/gradio_api/file={up_path}" style="width:100%;margin:6px 0;"></audio>
</div>'''
        else:
            upscale_section += '<div style="margin:6px 0;"><span style="display:inline-block;background:#7f1d1d;color:#ef4444;padding:2px 8px;border-radius:8px;font-size:0.75em;">Upscaled file missing</span></div>'

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

    return f"""<div style="border:1px solid #444;border-radius:12px;padding:16px;background:rgba(255,255,255,0.02);">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <h3 style="margin:0;font-size:1.1em;">{title}</h3>
    <span style="color:#888;font-size:0.8em;white-space:nowrap;margin-left:12px;">{ts}</span>
  </div>
  {f'<p style="color:#aaa;font-size:0.85em;margin:4px 0;">{desc}</p>' if desc else ''}
  {tag_pills}
  {audio_html}
  {upscale_section}
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
    for e in entries:
        audio_file = e.get("audio_file", "")
        upscaled_items = get_upscaled_files(e)

        # Prefer latest upscaled version for playlist playback
        preferred_path = None
        if upscaled_items:
            latest = upscaled_items[-1]
            fname = latest.get("file", "") if isinstance(latest, dict) else latest
            if fname:
                up_path = os.path.abspath(os.path.join(OUTPUT_DIR, fname))
                if os.path.isfile(up_path):
                    preferred_path = up_path
        if preferred_path is None:
            preferred_path = os.path.abspath(os.path.join(OUTPUT_DIR, audio_file))

        if os.path.isfile(preferred_path):
            tracks.append({
                "title": e.get("song_title", "Untitled"),
                "ts": e.get("timestamp", "")[:16].replace("T", " "),
                "src": f"/gradio_api/file={preferred_path}",
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

        # Music Generation Settings
        with gr.Accordion("Music Generation Settings", open=False):
            temperature = gr.Slider(0.1, 2.0, value=DEFAULT_GENERATION_PARAMS["temperature"], label="Temperature")
            cfg_scale = gr.Slider(0.0, 5.0, value=DEFAULT_GENERATION_PARAMS["cfg_scale"], label="CFG Scale")
            topk = gr.Slider(1, 200, value=DEFAULT_GENERATION_PARAMS["topk"], step=1, label="Top-K")
            max_length = gr.Slider(10, 240, value=DEFAULT_GENERATION_PARAMS["max_audio_length_ms"] // 1000, step=10, label="Max Length (seconds)")

        autoplay_cb = gr.Checkbox(label="Autoplay", value=True)

        # AudioSR Upscaling
        with gr.Accordion("AudioSR Upscaling", open=False):
            upscale_cb = gr.Checkbox(
                value=DEFAULT_AUDIOSR_ENABLED,
                label="Auto-upscale to 48kHz after generation",
                info="Uses AudioSR to upscale audio quality. Adds processing time. Requires ~1-2GB additional VRAM.",
            )
            upscale_format = gr.Dropdown(
                choices=AUDIOSR_FORMAT_CHOICES,
                value=AUDIOSR_FORMAT_LABELS.get(DEFAULT_AUDIOSR_FORMAT, "FLAC (Lossless)"),
                label="Output Format",
            )
            upscale_ddim = gr.Slider(
                10, 500, value=DEFAULT_AUDIOSR_DDIM_STEPS, step=10,
                label="DDIM Steps",
                info="Higher = better quality but slower (50 recommended)",
            )
            upscale_guidance = gr.Slider(
                1.0, 20.0, value=DEFAULT_AUDIOSR_GUIDANCE_SCALE, step=0.5,
                label="Guidance Scale",
                info="Classifier-free guidance strength (3.5 default)",
            )
            upscale_seed = gr.Number(
                value=DEFAULT_AUDIOSR_SEED,
                label="Seed",
                precision=0,
            )

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
                unload_audiosr_btn = gr.Button("Unload AudioSR", size="sm")
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
        music_inputs = [temperature, cfg_scale, topk, max_length, lazy_load_cb, model_variant, autoplay_cb,
                        upscale_cb, upscale_format, upscale_ddim, upscale_guidance, upscale_seed]

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

        def on_unload_audiosr():
            unload_audiosr()
            return _status_html("AudioSR model unloaded, GPU memory freed.", "success")

        unload_ollama_btn.click(on_unload_ollama, [llm_backend, ollama_url, ollama_model], [ollama_status])
        unload_btn.click(on_unload, [], [status_box])
        unload_audiosr_btn.click(on_unload_audiosr, [], [status_box])
        cancel_btn.click(lambda: (cancel_generation(), cancel_upscale()), [], [])

        # Clear button
        clear_btn.click(on_clear_all, [gen_desc_cb, gen_title_cb, gen_lyrics_cb, gen_tags_cb, song_desc, song_title_box, lyrics_box, tags_box], [song_desc, song_title_box, lyrics_box, tags_box, status_box])

    with gr.Tab("History") as history_tab:
        history_page = gr.State(value=0)
        playlist_player = gr.HTML(value="")
        history_status = gr.HTML(value="")
        page_info = gr.HTML(value="")

        # Fixed card slots (avoids @gr.render and its stale-handler bugs)
        _card_htmls = []
        _card_states = []
        _load_btns = []
        _upscale_btns = []
        _delete_btns = []
        for _i in range(HISTORY_PAGE_SIZE):
            _st = gr.State(value=None)
            _card_states.append(_st)
            _html = gr.HTML(visible=False)
            with gr.Row(visible=False) as _row:
                _lb = gr.Button("Load to Generator", size="sm", variant="primary", scale=1)
                _ub = gr.Button("Load to AudioSR", size="sm", variant="secondary", scale=1)
                _db = gr.Button("Delete", size="sm", variant="stop", scale=1)
            _card_htmls.append((_html, _row))
            _load_btns.append(_lb)
            _upscale_btns.append(_ub)
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

        # Wire up upscale handlers (loads song into AudioSR tab)
        def _slot_upscale(state):
            if not state:
                return gr.skip(), gr.skip(), _status_html("No entry selected.", "error")
            audio_file = state.get("audio_file", "")
            if not audio_file:
                return gr.skip(), gr.skip(), _status_html("No audio file found.", "error")
            audio_path = os.path.abspath(os.path.join(OUTPUT_DIR, audio_file))
            if not os.path.isfile(audio_path):
                return gr.skip(), gr.skip(), _status_html("Audio file not found.", "error")
            return audio_path, audio_file, _status_html(
                f"Loaded '{state.get('song_title', 'Untitled')}' into AudioSR tab.", "success"
            )

        _JS_SWITCH_TO_GENERATE = "()" + \
            " => document.querySelector('button[role=\"tab\"]').click()"

        for _i in range(HISTORY_PAGE_SIZE):
            _load_btns[_i].click(
                _slot_load,
                [_card_states[_i]],
                [song_desc, song_title_box, lyrics_box, tags_box, history_status],
            ).then(fn=None, js=_JS_SWITCH_TO_GENERATE)
            # Upscale buttons are wired after the AudioSR tab is defined (below)
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

    with gr.Tab("AudioSR Upscale"):
        gr.Markdown("### Upscale Audio to 48kHz")
        gr.Markdown("Upload any audio file and upscale it to 48kHz using AudioSR super-resolution.")

        upscale_source = gr.State(value=None)  # audio_file basename when loaded from History
        upload_audio = gr.Audio(label="Input Audio", type="filepath")

        with gr.Accordion("Upscale Settings", open=True):
            up_format = gr.Dropdown(
                choices=AUDIOSR_FORMAT_CHOICES,
                value=AUDIOSR_FORMAT_LABELS.get(DEFAULT_AUDIOSR_FORMAT, "FLAC (Lossless)"),
                label="Output Format",
            )
            up_ddim = gr.Slider(
                10, 500, value=DEFAULT_AUDIOSR_DDIM_STEPS, step=10,
                label="DDIM Steps",
                info="Higher = better quality but slower (50 recommended)",
            )
            up_guidance = gr.Slider(
                1.0, 20.0, value=DEFAULT_AUDIOSR_GUIDANCE_SCALE, step=0.5,
                label="Guidance Scale",
            )
            up_seed = gr.Number(value=DEFAULT_AUDIOSR_SEED, label="Seed", precision=0)

        with gr.Row():
            up_btn = gr.Button("Upscale", variant="primary", size="lg")
            up_cancel_btn = gr.Button("Cancel", variant="stop", size="lg", interactive=False)

        up_status = gr.HTML(value="")
        up_output = gr.Audio(label="Upscaled Audio (48kHz)", type="filepath", interactive=False)

        def on_standalone_upscale(audio_path, format_label, ddim_steps, guidance_scale, seed, source_file):
            if not audio_path:
                yield gr.skip(), _status_html("Please upload an audio file.", "error"), gr.update(interactive=True), gr.update(interactive=False), gr.skip()
                return

            # Verify source_file still matches the audio being upscaled
            if source_file:
                expected = os.path.abspath(os.path.join(OUTPUT_DIR, source_file))
                if os.path.abspath(audio_path) != expected:
                    source_file = None

            yield gr.skip(), _status_html("Upscaling audio to 48kHz...", "progress"), gr.update(interactive=False), gr.update(interactive=True), gr.skip()
            try:
                fmt = AUDIOSR_FORMAT_MAP.get(format_label, "flac")

                if source_file:
                    output_path = next_upscale_path(source_file, fmt)
                else:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    stem = os.path.splitext(os.path.basename(audio_path))[0]
                    output_path = os.path.join(OUTPUT_DIR, f"upscaled_{stem}_48kHz.{fmt}")

                result_path = upscale_audio(
                    input_path=audio_path,
                    output_path=output_path,
                    seed=int(seed),
                    ddim_steps=int(ddim_steps),
                    guidance_scale=guidance_scale,
                    output_format=fmt,
                )

                # Update history metadata if linked to a song
                if source_file:
                    json_path = os.path.splitext(os.path.join(OUTPUT_DIR, source_file))[0] + ".json"
                    existing = []
                    if os.path.isfile(json_path):
                        try:
                            with open(json_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            existing = get_upscaled_files(meta)
                        except (json.JSONDecodeError, OSError):
                            pass
                    new_entry = {
                        "file": os.path.basename(result_path),
                        "ddim_steps": int(ddim_steps),
                        "guidance_scale": guidance_scale,
                        "seed": int(seed),
                        "format": fmt,
                    }
                    update_generation(source_file, {"upscaled_files": existing + [new_entry]})

                yield result_path, _status_html("Upscaling complete! 48kHz audio ready.", "success"), gr.update(interactive=True), gr.update(interactive=False), None
            except UpscaleCancelled:
                yield gr.skip(), _status_html("Upscaling cancelled.", "info"), gr.update(interactive=True), gr.update(interactive=False), gr.skip()
            except Exception as e:
                logger.error("Standalone upscaling failed: %s", e)
                yield gr.skip(), _status_html(f"Error: {e}", "error"), gr.update(interactive=True), gr.update(interactive=False), gr.skip()

        up_btn.click(
            on_standalone_upscale,
            [upload_audio, up_format, up_ddim, up_guidance, up_seed, upscale_source],
            [up_output, up_status, up_btn, up_cancel_btn, upscale_source],
        )
        up_cancel_btn.click(lambda: cancel_upscale(), [], [])

    # Wire up History upscale buttons (needs upload_audio + upscale_source from AudioSR tab)
    _JS_SWITCH_TO_AUDIOSR = \
        "() => document.querySelectorAll('button[role=\"tab\"]')[2].click()"

    for _i in range(HISTORY_PAGE_SIZE):
        _upscale_btns[_i].click(
            _slot_upscale,
            [_card_states[_i]],
            [upload_audio, upscale_source, history_status],
        ).then(fn=None, js=_JS_SWITCH_TO_AUDIOSR)


if __name__ == "__main__":
    import sys
    share = "--share" in sys.argv
    from config import SERVER_HOST, SERVER_PORT
    app.launch(share=share, server_name=SERVER_HOST, server_port=SERVER_PORT, allowed_paths=[OUTPUT_DIR], js=PLAYLIST_JS)
