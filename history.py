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
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping corrupted history file %s: %s", fname, exc)
                continue
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries


def delete_generation(audio_file):
    """Delete a generation's audio and metadata files. Returns True on success."""
    if not audio_file:
        return False
    audio_path = os.path.realpath(os.path.join(OUTPUT_DIR, audio_file))
    # Prevent path traversal
    if not audio_path.startswith(os.path.realpath(OUTPUT_DIR) + os.sep):
        logger.warning("Path traversal attempt blocked: %s", audio_file)
        return False
    json_path = os.path.splitext(audio_path)[0] + ".json"
    deleted = False
    for p in (audio_path, json_path):
        if os.path.isfile(p):
            os.unlink(p)
            deleted = True
    return deleted
