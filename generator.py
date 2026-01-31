import os
import tempfile
import torch
from config import CKPT_DIR, OUTPUT_DIR

_pipeline = None


def get_pipeline(lazy_load=True):
    """Get or initialize the HeartMuLa pipeline."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    from model_manager import ensure_gen_config
    ensure_gen_config()
    from heartlib import HeartMuLaGenPipeline

    device_mula = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_codec = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _pipeline = HeartMuLaGenPipeline.from_pretrained(
        CKPT_DIR,
        device={"mula": device_mula, "codec": device_codec},
        dtype={"mula": torch.bfloat16, "codec": torch.float32},
        version="3B",
        lazy_load=lazy_load,
    )
    return _pipeline


def unload_pipeline():
    """Free GPU memory by unloading the pipeline."""
    global _pipeline
    _pipeline = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_models_downloaded():
    """Download models if not already present. Returns status message or None."""
    from model_manager import is_ready_for_generation, download_all_models
    if not is_ready_for_generation():
        download_all_models()
    if not is_ready_for_generation():
        raise RuntimeError("Failed to download required models. Check your internet connection.")


def generate_music(lyrics, tags, temperature=1.0, cfg_scale=1.5, topk=50,
                   max_audio_length_ms=240_000, lazy_load=True, output_path=None):
    """Generate music and return path to output file."""
    ensure_models_downloaded()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lyrics_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, dir=OUTPUT_DIR)
    lyrics_file.write(lyrics)
    lyrics_file.close()

    tags_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, dir=OUTPUT_DIR)
    tags_file.write(tags)
    tags_file.close()

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "output.mp3")

    pipe = get_pipeline(lazy_load=lazy_load)

    with torch.no_grad():
        pipe(
            {"lyrics": lyrics_file.name, "tags": tags_file.name},
            max_audio_length_ms=max_audio_length_ms,
            save_path=output_path,
            topk=topk,
            temperature=temperature,
            cfg_scale=cfg_scale,
        )

    os.unlink(lyrics_file.name)
    os.unlink(tags_file.name)

    return output_path
