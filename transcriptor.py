import gc
import logging
import threading

import torch

from config import CKPT_DIR

logger = logging.getLogger(__name__)

_transcriptor = None
_transcriptor_lock = threading.Lock()

TRANSCRIPTION_PARAMS = {
    "max_new_tokens": 256,
    "num_beams": 2,
    "task": "transcribe",
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.8,
    "temperature": (0.0, 0.1, 0.2, 0.4),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.4,
}


def get_transcriptor():
    """Lazy-load HeartTranscriptor pipeline on GPU (fp16) with CPU fallback."""
    global _transcriptor
    with _transcriptor_lock:
        if _transcriptor is not None:
            return _transcriptor
        logger.info("Loading HeartTranscriptor pipeline...")
        from heartlib import HeartTranscriptorPipeline

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            _transcriptor = HeartTranscriptorPipeline.from_pretrained(
                CKPT_DIR, device=device, dtype=dtype,
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM for transcriptor, falling back to CPU")
            device = torch.device("cpu")
            _transcriptor = HeartTranscriptorPipeline.from_pretrained(
                CKPT_DIR, device=device, dtype=torch.float32,
            )
        logger.info("HeartTranscriptor loaded on %s", device)
        return _transcriptor


def unload_transcriptor():
    """Free transcriptor model memory."""
    global _transcriptor
    with _transcriptor_lock:
        _transcriptor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("HeartTranscriptor unloaded")


def transcribe_audio(audio_path: str) -> str:
    """Transcribe lyrics from an audio file.

    Returns:
        Transcribed text string.
    """
    pipe = get_transcriptor()
    with torch.no_grad():
        result = pipe(audio_path, **TRANSCRIPTION_PARAMS)
    return result.get("text", "")
