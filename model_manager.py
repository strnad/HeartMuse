import os

os.environ["HF_HUB_CACHE"] = os.path.join(os.path.dirname(__file__), "ckpt")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "300"

from huggingface_hub import snapshot_download, hf_hub_download
from config import CKPT_DIR, REQUIRED_MODELS, HEARTMULGEN_REPO, HEARTMULGEN_FILES


def get_model_status():
    """Return list of (name, downloaded) for each required model."""
    result = []
    for model in REQUIRED_MODELS:
        local_path = os.path.join(CKPT_DIR, model["local_dir"])
        downloaded = os.path.isdir(local_path) and len(os.listdir(local_path)) > 0
        result.append((model["name"], downloaded))
    return result


def is_ready_for_generation():
    """Check if all required models are downloaded."""
    return all(downloaded for _, downloaded in get_model_status())


def download_all_models(progress=None):
    """Download all required models + config files."""
    ensure_gen_config()
    results = []
    total = len(REQUIRED_MODELS)
    for i, model in enumerate(REQUIRED_MODELS):
        if progress:
            progress((i, total), desc=f"Downloading {model['name']}...")
        local_path = os.path.join(CKPT_DIR, model["local_dir"])
        os.makedirs(local_path, exist_ok=True)
        try:
            snapshot_download(repo_id=model["repo_id"], local_dir=local_path)
            results.append(f"Downloaded {model['name']}")
        except Exception as e:
            results.append(f"Error downloading {model['name']}: {e}")
    return "\n".join(results)


def ensure_gen_config():
    """Download tokenizer.json and gen_config.json if missing."""
    missing = [f for f in HEARTMULGEN_FILES if not os.path.isfile(os.path.join(CKPT_DIR, f))]
    if not missing:
        return
    os.makedirs(CKPT_DIR, exist_ok=True)
    for filename in missing:
        hf_hub_download(HEARTMULGEN_REPO, filename, local_dir=CKPT_DIR)
