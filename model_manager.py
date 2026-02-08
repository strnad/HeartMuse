import logging
import os
import shutil

os.environ["HF_HUB_CACHE"] = os.path.join(os.path.dirname(__file__), "ckpt")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "300"

from huggingface_hub import snapshot_download, hf_hub_download
from config import (
    CKPT_DIR, MODEL_VARIANTS, CODEC_MODEL,
    HEARTMULGEN_REPO, HEARTMULGEN_FILES, DEFAULT_MODEL_VARIANT,
)

logger = logging.getLogger(__name__)


def _migrate_model_dirs():
    """Migrate old directory layout to new variant-aware layout.

    Previously the RL model was downloaded to HeartMuLa-oss-3B/ (the base model
    name). Rename it to HeartMuLa-oss-3B-RL/ so both variants can coexist.
    """
    old_path = os.path.join(CKPT_DIR, "HeartMuLa-oss-3B")
    new_path = os.path.join(CKPT_DIR, "HeartMuLa-oss-3B-RL")

    if (os.path.isdir(old_path)
            and not os.path.islink(old_path)
            and not os.path.exists(new_path)):
        logger.info("Migrating RL model directory: %s -> %s", old_path, new_path)
        shutil.move(old_path, new_path)


# Run migration on import
_migrate_model_dirs()


def _is_model_downloaded(local_dir):
    """Check if a model directory exists and has files."""
    path = os.path.join(CKPT_DIR, local_dir)
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def get_model_status(variant=None):
    """Return list of (name, downloaded) for the selected variant + codec.

    Args:
        variant: Variant key ("rl", "base") or None for default.
    """
    if variant is None:
        variant = DEFAULT_MODEL_VARIANT
    v = MODEL_VARIANTS.get(variant, MODEL_VARIANTS["rl"])

    return [
        (v["name"], _is_model_downloaded(v["local_dir"])),
        (CODEC_MODEL["name"], _is_model_downloaded(CODEC_MODEL["local_dir"])),
    ]


def is_ready_for_generation(variant=None):
    """Check if the selected variant + codec + config files are downloaded."""
    if variant is None:
        variant = DEFAULT_MODEL_VARIANT

    # Check config files
    for f in HEARTMULGEN_FILES:
        if not os.path.isfile(os.path.join(CKPT_DIR, f)):
            return False

    # Check variant model + codec
    v = MODEL_VARIANTS.get(variant, MODEL_VARIANTS["rl"])
    if not _is_model_downloaded(v["local_dir"]):
        return False
    if not _is_model_downloaded(CODEC_MODEL["local_dir"]):
        return False

    return True


def download_all_models(variant=None):
    """Download the selected variant + codec + config files.

    Args:
        variant: Variant key ("rl", "base") or None for default.
    """
    if variant is None:
        variant = DEFAULT_MODEL_VARIANT

    ensure_gen_config()

    v = MODEL_VARIANTS.get(variant, MODEL_VARIANTS["rl"])
    results = []

    # Download variant model
    local_path = os.path.join(CKPT_DIR, v["local_dir"])
    os.makedirs(local_path, exist_ok=True)
    try:
        snapshot_download(repo_id=v["repo_id"], local_dir=local_path)
        results.append(f"Downloaded {v['name']}")
        logger.info("Downloaded %s", v['name'])
    except Exception as e:
        logger.error("Failed to download %s: %s", v['name'], e)
        results.append(f"Error downloading {v['name']}: {e}")

    # Download codec
    codec_path = os.path.join(CKPT_DIR, CODEC_MODEL["local_dir"])
    os.makedirs(codec_path, exist_ok=True)
    try:
        snapshot_download(repo_id=CODEC_MODEL["repo_id"], local_dir=codec_path)
        results.append(f"Downloaded {CODEC_MODEL['name']}")
        logger.info("Downloaded %s", CODEC_MODEL['name'])
    except Exception as e:
        logger.error("Failed to download %s: %s", CODEC_MODEL['name'], e)
        results.append(f"Error downloading {CODEC_MODEL['name']}: {e}")

    return "\n".join(results)


def ensure_gen_config():
    """Download tokenizer.json and gen_config.json if missing."""
    missing = [f for f in HEARTMULGEN_FILES if not os.path.isfile(os.path.join(CKPT_DIR, f))]
    if not missing:
        return
    os.makedirs(CKPT_DIR, exist_ok=True)
    for filename in missing:
        hf_hub_download(HEARTMULGEN_REPO, filename, local_dir=CKPT_DIR)
