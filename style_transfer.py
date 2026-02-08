import gc
import logging
import threading

import torch

logger = logging.getLogger(__name__)

_muq_model = None
_muq_lock = threading.Lock()


def get_muq_model():
    """Lazy-load MuQ-MuLan on CPU (avoids GPU contention with HeartMuLa)."""
    global _muq_model
    with _muq_lock:
        if _muq_model is not None:
            return _muq_model
        logger.info("Loading MuQ-MuLan model...")
        from muq import MuQMuLan
        _muq_model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
        _muq_model = _muq_model.to("cpu").float().eval()
        logger.info("MuQ-MuLan loaded on CPU")
        return _muq_model


def unload_muq():
    """Free MuQ model memory."""
    global _muq_model
    with _muq_lock:
        _muq_model = None
        gc.collect()
        logger.info("MuQ-MuLan unloaded")


def extract_style_embedding(audio_path: str) -> torch.Tensor:
    """Extract 512D style embedding from a reference audio file.

    MuQ-MuLan requires 24 kHz mono input and fp32 precision.

    Returns:
        torch.Tensor of shape [512] in bfloat16 (matches HeartMuLa dtype).
    """
    import librosa

    model = get_muq_model()
    wav, _sr = librosa.load(audio_path, sr=24000)
    wavs = torch.tensor(wav).unsqueeze(0).float()  # [1, samples]
    with torch.no_grad():
        embedding = model(wavs=wavs)  # [1, 512]
    return embedding.squeeze(0).to(torch.bfloat16)  # [512]
