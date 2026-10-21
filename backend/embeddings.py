"""
Optional CLIP embeddings (Hugging Face transformers).

Used for
  * semantic re-ranking: "a woman with a stroller" — anything YOLO's class
    list cannot express is matched against the crop of every track
  * entity merging: two tracks that look the same are the same person

Enabled with   CLIP_RERANK=1   (default: on when transformers + torch are
installed). The first call downloads ~600 MB of weights.
"""

import os
import threading
from typing import Optional

import numpy as np

CLIP_MODEL = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
_ENABLED = os.environ.get("CLIP_RERANK", "1") not in ("0", "false", "no")
_lock = threading.Lock()
_model = None
_processor = None
_failed: Optional[str] = None


def error() -> Optional[str]:
    """Why CLIP could not be loaded (None when fine or not tried yet)."""
    return _failed


def available() -> bool:
    """True when CLIP can be used (does not load the weights yet)."""
    if not _ENABLED or _failed:
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _load():
    global _model, _processor, _failed
    with _lock:
        if _model is not None or _failed:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            # safetensors: no torch.load, works with every torch version.
            # Cached weights first (no network round-trips), download otherwise.
            try:
                _model = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True,
                                                   local_files_only=True)
                _processor = CLIPProcessor.from_pretrained(CLIP_MODEL, local_files_only=True)
            except OSError:
                _model = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True)
                _processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            _model.eval()
        except Exception as e:  # noqa: BLE001
            _failed = f"{type(e).__name__}: {e}"


def _norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def embed_images(crops_bgr: list) -> Optional[np.ndarray]:
    """[HxWx3 BGR uint8, ...] -> (N, D) L2-normalised float32, or None."""
    if not available() or not crops_bgr:
        return None
    _load()
    if _model is None:
        return None
    import torch
    from PIL import Image
    out = []
    with torch.no_grad():          # grad mode is per thread -> set it here
        for i in range(0, len(crops_bgr), 32):
            batch = [Image.fromarray(c[:, :, ::-1]) for c in crops_bgr[i:i + 32]]
            inputs = _processor(images=batch, return_tensors="pt")
            feats = _model.get_image_features(**inputs)
            out.append(feats.detach().cpu().numpy())
    return _norm(np.concatenate(out))


def embed_text(text: str) -> Optional[np.ndarray]:
    if not available() or not text.strip():
        return None
    _load()
    if _model is None:
        return None
    import torch
    with torch.no_grad():
        inputs = _processor(text=[text], return_tensors="pt", padding=True,
                            truncation=True)
        feats = _model.get_text_features(**inputs)
    return _norm(feats.detach().cpu().numpy())[0]


def to_blob(vec: Optional[np.ndarray]) -> Optional[bytes]:
    return None if vec is None else vec.astype(np.float32).tobytes()


def from_blob(blob: Optional[bytes]) -> Optional[np.ndarray]:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32)
