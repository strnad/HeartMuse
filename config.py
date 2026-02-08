import os
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _env_float(key, default):
    """Read a float from env, falling back to default on invalid values."""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        logger.warning("Invalid value for %s, using default %s", key, default)
        return float(default)


def _env_int(key, default):
    """Read an int from env, falling back to default on invalid values."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        logger.warning("Invalid value for %s, using default %s", key, default)
        return int(default)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(BASE_DIR, "ckpt"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, "output"))

HEARTMULGEN_REPO = "HeartMuLa/HeartMuLaGen"
HEARTMULGEN_FILES = ["tokenizer.json", "gen_config.json"]

# --- Model Variants ---
# Each variant has its own HuggingFace repo and local directory.
# The "version" string is passed to heartlib's from_pretrained() which constructs
# the model path as: CKPT_DIR/HeartMuLa-oss-{version}/
MODEL_VARIANTS = {
    "rl": {
        "name": "HeartMuLa 3B RL",
        "repo_id": "HeartMuLa/HeartMuLa-RL-oss-3B-20260123",
        "local_dir": "HeartMuLa-oss-3B-RL",
        "version": "3B-RL",
    },
    "base": {
        "name": "HeartMuLa 3B",
        "repo_id": "HeartMuLa/HeartMuLa-oss-3B",
        "local_dir": "HeartMuLa-oss-3B",
        "version": "3B",
    },
}

CODEC_MODEL = {
    "name": "HeartCodec",
    "repo_id": "HeartMuLa/HeartCodec-oss-20260123",
    "local_dir": "HeartCodec-oss",
}

MODEL_VARIANT_LABELS = {
    "rl": "HeartMuLa 3B RL (Recommended)",
    "base": "HeartMuLa 3B (Base)",
}

DEFAULT_MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "rl")

DEFAULT_GENERATION_PARAMS = {
    "temperature": _env_float("MUSIC_TEMPERATURE", "1.0"),
    "cfg_scale": _env_float("MUSIC_CFG_SCALE", "1.5"),
    "topk": _env_int("MUSIC_TOPK", "50"),
    "max_audio_length_ms": _env_int("MUSIC_MAX_LENGTH_SEC", "240") * 1000,
}

DEFAULT_NUM_VARIANTS = _env_int("MUSIC_NUM_VARIANTS", "1")

DEFAULT_LAZY_LOAD = os.environ.get("LAZY_LOAD", "true").lower() in ("1", "true", "yes")

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "glm-4.7-flash")

DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_OPENAI_URL = os.environ.get("OPENAI_URL", "https://api.openai.com/v1")
DEFAULT_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_OPENAI_MODELS = [
    m.strip()
    for m in os.environ.get(
        "OPENAI_MODELS", "gpt-4o-mini,gpt-4o,gpt-4.1-mini,gpt-4.1,o4-mini,gpt-5-mini,gpt-5,gpt-5.2"
    ).split(",")
    if m.strip()
]

DEFAULT_LLM_BACKEND = os.environ.get("LLM_BACKEND", "Ollama")
DEFAULT_LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", "0.7")
DEFAULT_LLM_TIMEOUT = _env_int("LLM_TIMEOUT", "120")

STYLE_TRANSFER_ENABLED = os.environ.get("STYLE_TRANSFER", "true").lower() in ("1", "true", "yes")

TRANSCRIPTION_ENABLED = os.environ.get("TRANSCRIPTION", "true").lower() in ("1", "true", "yes")

TRANSCRIPTOR_MODEL = {
    "name": "HeartTranscriptor",
    "repo_id": "HeartMuLa/HeartTranscriptor-oss",
    "local_dir": "HeartTranscriptor-oss",
}

SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = _env_int("SERVER_PORT", "7860")
