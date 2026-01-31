import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(BASE_DIR, "ckpt"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, "output"))

HEARTMULGEN_REPO = "HeartMuLa/HeartMuLaGen"
HEARTMULGEN_FILES = ["tokenizer.json", "gen_config.json"]

REQUIRED_MODELS = [
    {
        "name": "HeartMuLa 3B RL",
        "repo_id": "HeartMuLa/HeartMuLa-RL-oss-3B-20260123",
        "local_dir": "HeartMuLa-oss-3B",
    },
    {
        "name": "HeartCodec",
        "repo_id": "HeartMuLa/HeartCodec-oss-20260123",
        "local_dir": "HeartCodec-oss",
    },
]

DEFAULT_GENERATION_PARAMS = {
    "temperature": float(os.environ.get("MUSIC_TEMPERATURE", "1.0")),
    "cfg_scale": float(os.environ.get("MUSIC_CFG_SCALE", "1.5")),
    "topk": int(os.environ.get("MUSIC_TOPK", "50")),
    "max_audio_length_ms": int(os.environ.get("MUSIC_MAX_LENGTH_SEC", "240")) * 1000,
}

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "glm-4.7-flash")

DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_OPENAI_URL = os.environ.get("OPENAI_URL", "https://api.openai.com/v1")
DEFAULT_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_OPENAI_MODELS = [
    m.strip()
    for m in os.environ.get(
        "OPENAI_MODELS", "gpt-4o-mini,gpt-4o,gpt-4.1-mini,gpt-4.1,o4-mini"
    ).split(",")
    if m.strip()
]

DEFAULT_LLM_BACKEND = os.environ.get("LLM_BACKEND", "Ollama")
DEFAULT_LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
DEFAULT_LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))
