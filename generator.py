import gc
import logging
import os
import tempfile
import threading
import weakref
import torch
from config import (
    CKPT_DIR, OUTPUT_DIR, DEFAULT_LAZY_LOAD, DEFAULT_MODEL_VARIANT,
    MODEL_VARIANTS,
)

logger = logging.getLogger(__name__)

# Suppress harmless torchtune warning when setup_caches() is called on a model
# whose KV caches are already allocated (e.g. after cancel + re-generate).
# The caches are properly reused; the warning is just noise.
logging.getLogger("torchtune.modules.attention").setLevel(logging.ERROR)

_pipeline = None
_pipeline_lock = threading.Lock()
_active_lazy_load = None  # tracks the lazy_load setting the current pipeline was loaded with
_active_variant = None    # tracks the model variant the current pipeline was loaded with
_cancel_event = threading.Event()


class GenerationCancelled(Exception):
    """Raised when user cancels music generation."""
    pass


def cancel_generation():
    """Signal the current generation to stop."""
    _cancel_event.set()


def is_generation_cancelled():
    """Check if cancellation has been requested."""
    return _cancel_event.is_set()


def get_pipeline(lazy_load=False, model_variant=None):
    """Get or initialize the HeartMuLa pipeline.

    If the pipeline is already loaded with a different variant or lazy_load
    setting, it is unloaded first.
    """
    global _pipeline, _active_lazy_load, _active_variant

    if model_variant is None:
        model_variant = DEFAULT_MODEL_VARIANT

    variant = MODEL_VARIANTS.get(model_variant, MODEL_VARIANTS["rl"])

    with _pipeline_lock:
        if _pipeline is not None and (_active_lazy_load != lazy_load or _active_variant != model_variant):
            logger.info("Settings changed (lazy_load: %s->%s, variant: %s->%s), reloading pipeline",
                        _active_lazy_load, lazy_load, _active_variant, model_variant)
            _pipeline = None
            _force_gpu_cleanup()

        if _pipeline is not None:
            return _pipeline

        version = variant["version"]

        logger.info("Initializing HeartMuLa pipeline (variant=%s, version=%s, lazy_load=%s)",
                     model_variant, version, lazy_load)

        from model_manager import ensure_gen_config
        ensure_gen_config()
        from heartlib import HeartMuLaGenPipeline

        device_mula = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_codec = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            _pipeline = HeartMuLaGenPipeline.from_pretrained(
                CKPT_DIR,
                device={"mula": device_mula, "codec": device_codec},
                dtype={"mula": torch.bfloat16, "codec": torch.float32},
                version=version,
                lazy_load=lazy_load,
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA out of memory, falling back to CPU")
            _pipeline = HeartMuLaGenPipeline.from_pretrained(
                CKPT_DIR,
                device={"mula": torch.device("cpu"), "codec": torch.device("cpu")},
                dtype={"mula": torch.bfloat16, "codec": torch.float32},
                version=version,
                lazy_load=lazy_load,
            )

        _active_lazy_load = lazy_load
        _active_variant = model_variant
        logger.info("Pipeline initialized on %s", device_mula)
        return _pipeline


def _force_gpu_cleanup():
    """Force cleanup of GPU memory (cyclic refs, CUDA cache)."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def unload_pipeline():
    """Free GPU memory by unloading the pipeline."""
    global _pipeline, _active_lazy_load, _active_variant
    with _pipeline_lock:
        _pipeline = None
        _active_lazy_load = None
        _active_variant = None
        _force_gpu_cleanup()
        logger.info("Pipeline unloaded")


def ensure_models_downloaded(model_variant=None):
    """Download models if not already present."""
    from model_manager import is_ready_for_generation, download_all_models
    if not is_ready_for_generation(model_variant):
        download_all_models(model_variant)
    if not is_ready_for_generation(model_variant):
        raise RuntimeError("Failed to download required models. Check your internet connection.")


def generate_music(lyrics, tags, temperature=1.0, cfg_scale=1.5, topk=50,
                   max_audio_length_ms=240_000, output_path=None,
                   lazy_load=None, model_variant=None, style_embedding=None,
                   seed=None):
    """Generate music and return path to output file."""
    if lazy_load is None:
        lazy_load = DEFAULT_LAZY_LOAD
    if model_variant is None:
        model_variant = DEFAULT_MODEL_VARIANT

    ensure_models_downloaded(model_variant)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lyrics_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, dir=OUTPUT_DIR, encoding="utf-8")
    lyrics_file.write(lyrics)
    lyrics_file.close()

    tags_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, dir=OUTPUT_DIR, encoding="utf-8")
    tags_file.write(tags)
    tags_file.close()

    _cancel_event.clear()
    pipe = None
    try:
        if output_path is None:
            output_path = os.path.join(OUTPUT_DIR, "output.mp3")

        pipe = get_pipeline(lazy_load=lazy_load, model_variant=model_variant)

        # Install style embedding monkeypatch (inject real MuQ embedding instead of zeros)
        if style_embedding is not None:
            pipe._style_embedding = style_embedding.to(pipe.mula_device)
            _orig_preprocess = pipe.preprocess.__func__

            def _patched_preprocess(self, inputs, cfg_scale):
                result = _orig_preprocess(self, inputs, cfg_scale)
                if hasattr(self, '_style_embedding') and self._style_embedding is not None:
                    emb = self._style_embedding
                    result["muq_embed"][:] = emb.unsqueeze(0).expand_as(result["muq_embed"])
                return result

            pipe.preprocess = _patched_preprocess.__get__(pipe, type(pipe))

        mula_model = pipe.mula

        # Install cancellation hook using weak references so that heartlib's
        # _unload() can actually free the model from VRAM between stages.
        # Strong references (bound methods, local vars) would prevent GC.
        _mula_ref = weakref.ref(mula_model)
        _orig_generate_frame = type(mula_model).generate_frame

        def cancellable_generate_frame(*args, **kwargs):
            if _cancel_event.is_set():
                raise GenerationCancelled()
            model = _mula_ref()
            if model is None:
                raise GenerationCancelled()
            return _orig_generate_frame(model, *args, **kwargs)

        mula_model.generate_frame = cancellable_generate_frame
        del mula_model  # release strong reference; only pipe._mula remains

        try:
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            with torch.no_grad():
                pipe(
                    {"lyrics": lyrics_file.name, "tags": tags_file.name},
                    max_audio_length_ms=max_audio_length_ms,
                    save_path=output_path,
                    topk=topk,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                )
            return output_path
        except Exception:
            _force_gpu_cleanup()
            raise
        finally:
            model = _mula_ref()
            if model is not None and 'generate_frame' in model.__dict__:
                del model.generate_frame
            del model
    finally:
        # Clean up style embedding monkeypatch
        if pipe is not None:
            if hasattr(pipe, '_style_embedding'):
                del pipe._style_embedding
            if 'preprocess' in pipe.__dict__:
                del pipe.preprocess  # restore original class method

        _cancel_event.clear()
        for f in (lyrics_file.name, tags_file.name):
            try:
                os.unlink(f)
            except OSError:
                pass
