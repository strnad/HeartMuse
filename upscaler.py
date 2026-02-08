import gc
import logging
import os
import tempfile
import threading

import torch
import torchaudio

from config import (
    OUTPUT_DIR,
    DEFAULT_AUDIOSR_DDIM_STEPS,
    DEFAULT_AUDIOSR_GUIDANCE_SCALE,
    DEFAULT_AUDIOSR_FORMAT,
)

logger = logging.getLogger(__name__)

AUDIOSR_NOT_INSTALLED_MSG = (
    "AudioSR is not installed. "
    "Run ./install_audiosr.sh (Linux/macOS) or install_audiosr.bat (Windows) to install it."
)


def is_audiosr_available():
    """Check if the audiosr package is installed."""
    try:
        import audiosr  # noqa: F401
        return True
    except ImportError:
        return False


_audiosr_model = None
_audiosr_lock = threading.Lock()
_cancel_event = threading.Event()


class UpscaleCancelled(Exception):
    """Raised when user cancels upscaling."""
    pass


def cancel_upscale():
    """Signal the current upscale to stop."""
    _cancel_event.set()


def _force_gpu_cleanup():
    """Force cleanup of GPU memory."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def get_audiosr_model():
    """Get or initialize the AudioSR model, unloading HeartMuLa first to free VRAM."""
    global _audiosr_model

    with _audiosr_lock:
        if _audiosr_model is not None:
            return _audiosr_model

        if not is_audiosr_available():
            raise RuntimeError(AUDIOSR_NOT_INSTALLED_MSG)

        # Unload HeartMuLa pipeline to free GPU memory
        from generator import unload_pipeline
        unload_pipeline()

        logger.info("Loading AudioSR model...")
        from audiosr import build_model
        _audiosr_model = build_model(model_name="basic", device="auto")
        logger.info("AudioSR model loaded")
        return _audiosr_model


def unload_audiosr():
    """Free GPU memory by unloading the AudioSR model."""
    global _audiosr_model
    with _audiosr_lock:
        if _audiosr_model is not None:
            try:
                _audiosr_model.cpu()
            except Exception:
                pass
            del _audiosr_model
        _audiosr_model = None
        _force_gpu_cleanup()
        logger.info("AudioSR model unloaded")


def _upscale_file(model, input_path, duration_s, seed, ddim_steps, guidance_scale):
    """Upscale a single (mono) audio file and return a CPU tensor [1, samples]."""
    if duration_s > 10.0:
        logger.info("Long audio (%.1fs), using chunked processing", duration_s)
        from audiosr import super_resolution_long_audio
        waveform = super_resolution_long_audio(
            model,
            input_file=input_path,
            seed=seed,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
        )
        # Returns [channels, samples] directly (no batch dim)
        audio_data = waveform
    else:
        from audiosr import super_resolution
        waveform = super_resolution(
            model,
            input_file=input_path,
            seed=seed,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
        )
        # Returns [batch, channels, samples]
        audio_data = waveform[0]

    del waveform

    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.to(torch.float32).cpu()
    else:
        audio_data = torch.from_numpy(audio_data).to(torch.float32)

    if audio_data.dim() == 1:
        audio_data = audio_data.unsqueeze(0)

    return audio_data


def upscale_audio(input_path, output_path=None, seed=42, ddim_steps=None,
                  guidance_scale=None, output_format=None):
    """Upscale an audio file to 48kHz using AudioSR.

    Stereo files are split into separate channels, upscaled individually,
    and recombined to preserve stereo imaging.

    Args:
        input_path: Path to input audio file (MP3/WAV/FLAC/etc.)
        output_path: Path for output file. If None, auto-generated from input_path.
        seed: Random seed for reproducibility.
        ddim_steps: Number of DDIM diffusion steps (default from config).
        guidance_scale: Classifier-free guidance scale (default from config).
        output_format: Output format - 'flac', 'wav', or 'mp3' (default from config).

    Returns:
        Path to the upscaled audio file.
    """
    if ddim_steps is None:
        ddim_steps = DEFAULT_AUDIOSR_DDIM_STEPS
    if guidance_scale is None:
        guidance_scale = DEFAULT_AUDIOSR_GUIDANCE_SCALE
    if output_format is None:
        output_format = DEFAULT_AUDIOSR_FORMAT

    ext = f".{output_format}"
    if output_path is None:
        stem = os.path.splitext(input_path)[0]
        output_path = f"{stem}_48kHz{ext}"

    _cancel_event.clear()

    try:
        model = get_audiosr_model()

        if _cancel_event.is_set():
            raise UpscaleCancelled()

        info = torchaudio.info(input_path)
        duration_s = info.num_frames / info.sample_rate
        num_channels = info.num_channels

        if num_channels >= 2:
            # Stereo: split channels, upscale each separately, recombine
            waveform, sr = torchaudio.load(input_path)
            temp_files = []
            try:
                channels_upscaled = []
                for ch in range(2):
                    if _cancel_event.is_set():
                        raise UpscaleCancelled()

                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".wav", dir=OUTPUT_DIR, delete=False,
                    )
                    tmp.close()
                    temp_files.append(tmp.name)
                    torchaudio.save(tmp.name, waveform[ch : ch + 1], sr)

                    logger.info("Upscaling channel %d/2", ch + 1)
                    ch_data = _upscale_file(
                        model, tmp.name, duration_s, seed, ddim_steps, guidance_scale,
                    )
                    channels_upscaled.append(ch_data[0])  # [samples]

                del waveform
                _force_gpu_cleanup()

                audio_data = torch.stack(channels_upscaled, dim=0)  # [2, samples]
                torchaudio.save(output_path, audio_data, 48000)
                logger.info("Upscaled stereo audio saved to %s", output_path)
                return output_path
            finally:
                for f in temp_files:
                    try:
                        os.unlink(f)
                    except OSError:
                        pass
        else:
            # Mono: upscale directly
            audio_data = _upscale_file(
                model, input_path, duration_s, seed, ddim_steps, guidance_scale,
            )
            _force_gpu_cleanup()
            torchaudio.save(output_path, audio_data, 48000)
            logger.info("Upscaled audio saved to %s", output_path)
            return output_path

    except UpscaleCancelled:
        raise
    except Exception:
        _force_gpu_cleanup()
        raise
    finally:
        _cancel_event.clear()
