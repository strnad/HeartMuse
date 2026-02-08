import logging
import os
import re
import json
import tempfile
from datetime import datetime
from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def sanitize_filename(title, max_len=50):
    """Convert a song title to a safe filename component."""
    s = re.sub(r'[^\w\s-]', '', title).strip()
    s = re.sub(r'[\s]+', '_', s)
    return s[:max_len] if s else "Untitled"


def generate_output_path(song_title):
    """Generate a timestamped output path like YYYYMMDD_HHMMSS_Title.mp3."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = sanitize_filename(song_title)
    return os.path.join(OUTPUT_DIR, f"{ts}_{safe}.mp3")


def save_generation(song_title, description, lyrics, tags, params, audio_path):
    """Save generation metadata as JSON alongside the audio file."""
    meta = {
        "song_title": song_title,
        "description": description,
        "lyrics": lyrics,
        "tags": tags,
        "parameters": params,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "audio_file": os.path.basename(audio_path),
    }
    json_path = os.path.splitext(audio_path)[0] + ".json"
    fd, tmp = tempfile.mkstemp(dir=OUTPUT_DIR, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp, json_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return json_path


def update_generation(audio_file, updates):
    """Update fields in a generation's JSON metadata.

    Args:
        audio_file: The audio filename (basename) to identify the generation.
        updates: Dict of fields to update/add to the metadata.
    Returns:
        True on success, False on failure.
    """
    if not audio_file:
        return False
    audio_path = os.path.realpath(os.path.join(OUTPUT_DIR, audio_file))
    if not audio_path.startswith(os.path.realpath(OUTPUT_DIR) + os.sep):
        logger.warning("Path traversal attempt blocked: %s", audio_file)
        return False
    json_path = os.path.splitext(audio_path)[0] + ".json"
    if not os.path.isfile(json_path):
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta.update(updates)
        fd, tmp = tempfile.mkstemp(dir=OUTPUT_DIR, suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            os.replace(tmp, json_path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return True
    except Exception:
        logger.error("Failed to update metadata for %s", audio_file)
        return False


def load_history():
    """Load all generation metadata, sorted newest first."""
    if not os.path.isdir(OUTPUT_DIR):
        return []
    entries = []
    for fname in os.listdir(OUTPUT_DIR):
        if fname.endswith(".json"):
            path = os.path.join(OUTPUT_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entries.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries


def get_upscaled_files(entry):
    """Return a normalized list of upscaled file entries from a history entry.

    Handles both old format (single ``upscaled_file``) and new format
    (``upscaled_files`` list).  Each item is a dict with at least a ``file`` key.
    """
    # New format
    items = entry.get("upscaled_files")
    if isinstance(items, list) and items:
        return items

    # Old format – single file
    old = entry.get("upscaled_file")
    if old:
        params = entry.get("upscale_params") or {}
        return [{"file": old, **params}]

    return []


def next_upscale_path(audio_file, fmt):
    """Return the next available upscaled filename for *audio_file*.

    Naming scheme:
      1st  → {stem}_48kHz.{fmt}
      2nd  → {stem}_48kHz_2.{fmt}
      3rd  → {stem}_48kHz_3.{fmt}  …
    """
    stem = os.path.splitext(audio_file)[0]
    ext = f".{fmt}"
    candidate = os.path.join(OUTPUT_DIR, f"{stem}_48kHz{ext}")
    if not os.path.exists(candidate):
        return candidate
    n = 2
    while True:
        candidate = os.path.join(OUTPUT_DIR, f"{stem}_48kHz_{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1


def delete_generation(audio_file):
    """Delete a generation's audio, metadata, and upscaled files. Returns True on success."""
    if not audio_file:
        return False
    audio_path = os.path.realpath(os.path.join(OUTPUT_DIR, audio_file))
    # Prevent path traversal
    if not audio_path.startswith(os.path.realpath(OUTPUT_DIR) + os.sep):
        logger.warning("Path traversal attempt blocked: %s", audio_file)
        return False
    json_path = os.path.splitext(audio_path)[0] + ".json"

    # Collect all upscaled file paths from metadata
    upscaled_paths = []
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for item in get_upscaled_files(meta):
                fname = item.get("file") if isinstance(item, dict) else item
                if fname:
                    up = os.path.realpath(os.path.join(OUTPUT_DIR, fname))
                    if up.startswith(os.path.realpath(OUTPUT_DIR) + os.sep):
                        upscaled_paths.append(up)
        except (json.JSONDecodeError, OSError):
            pass

    deleted = False
    for p in [audio_path, json_path] + upscaled_paths:
        if os.path.isfile(p):
            os.unlink(p)
            deleted = True
    return deleted
