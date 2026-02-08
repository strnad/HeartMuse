import os
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

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
    "temperature": float(os.environ.get("MUSIC_TEMPERATURE", "1.0")),
    "cfg_scale": float(os.environ.get("MUSIC_CFG_SCALE", "1.5")),
    "topk": int(os.environ.get("MUSIC_TOPK", "50")),
    "max_audio_length_ms": int(os.environ.get("MUSIC_MAX_LENGTH_SEC", "240")) * 1000,
}

DEFAULT_LAZY_LOAD = os.environ.get("LAZY_LOAD", "true").lower() in ("1", "true", "yes")

# --- AudioSR Upscaling ---
DEFAULT_AUDIOSR_ENABLED = os.environ.get("AUDIOSR_ENABLED", "false").lower() in ("1", "true", "yes")
DEFAULT_AUDIOSR_DDIM_STEPS = int(os.environ.get("AUDIOSR_DDIM_STEPS", "50"))
DEFAULT_AUDIOSR_GUIDANCE_SCALE = float(os.environ.get("AUDIOSR_GUIDANCE_SCALE", "3.5"))
DEFAULT_AUDIOSR_SEED = int(os.environ.get("AUDIOSR_SEED", "42"))
DEFAULT_AUDIOSR_FORMAT = os.environ.get("AUDIOSR_FORMAT", "mp3").lower()
AUDIOSR_FORMAT_CHOICES = ["FLAC (Lossless)", "WAV (Uncompressed)", "MP3"]
AUDIOSR_FORMAT_MAP = {"FLAC (Lossless)": "flac", "WAV (Uncompressed)": "wav", "MP3": "mp3"}
AUDIOSR_FORMAT_LABELS = {v: k for k, v in AUDIOSR_FORMAT_MAP.items()}

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
DEFAULT_LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
DEFAULT_LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))

SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "7860"))
