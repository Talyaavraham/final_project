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
_device = "cpu"
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
    global _model, _processor, _device, _failed
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
            # GPU when there is one (Colab); float32 either way, so the vectors match a CPU index
            _device = os.environ.get("CLIP_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
            _model.to(_device)
            _model.eval()
        except Exception as e:  # noqa: BLE001
            _failed = f"{type(e).__name__}: {e}"


def _norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def _tensor(out):
    """get_image/text_features: a tensor in transformers 4.x; in 5.x an output object whose
    pooler_output holds the projected embedding."""
    return out if hasattr(out, "detach") else out.pooler_output


def _pil(crop_bgr):
    """BGR uint8 -> PIL RGB, at least 8x8 (a 1 px sliver is useless and breaks the processor)."""
    from PIL import Image
    img = Image.fromarray(np.ascontiguousarray(crop_bgr[:, :, ::-1]))
    if img.width < 8 or img.height < 8:
        img = img.resize((max(img.width, 8), max(img.height, 8)))
    return img


def embed_images(crops_bgr: list) -> Optional[np.ndarray]:
    """[HxWx3 BGR uint8, ...] -> (N, D) L2-normalised float32, or None."""
    if not available() or not crops_bgr:
        return None
    _load()
    if _model is None:
        return None
    import torch
    out = []
    with torch.no_grad():          # grad mode is per thread -> set it here
        for i in range(0, len(crops_bgr), 32):
            batch = [_pil(c) for c in crops_bgr[i:i + 32]]
            # channels_last: a 1- or 3-pixel-high crop is otherwise read as channels-first
            inputs = _processor(images=batch, return_tensors="pt", input_data_format="channels_last").to(_device)
            feats = _tensor(_model.get_image_features(**inputs))
            out.append(feats.detach().cpu().numpy())
    return _norm(np.concatenate(out))


_text_cache: dict = {}       # text -> raw CLIP vector; the same prompts are asked again and again

# Text-side adapter (train/clip_adapter.py): t -> norm(t + t @ W + b), trained on UCA
# sentences so that a description lands nearer the CCTV frames it describes. Image
# vectors are untouched, so an existing index keeps working. CLIP_ADAPTER=0 turns it off.
ADAPTER_PATH = os.environ.get("CLIP_ADAPTER",
                              os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_adapter.npz"))
_adapter = None              # (W, b), False when there is none


def adapter():
    """-> (W, b) or None."""
    global _adapter
    if _adapter is None:
        _adapter = False
        if ADAPTER_PATH not in ("", "0", "false", "no") and os.path.isfile(ADAPTER_PATH):
            z = np.load(ADAPTER_PATH)
            _adapter = (z["W"].astype(np.float32), z["b"].astype(np.float32))
    return _adapter or None


def set_adapter(path: Optional[str]):
    """Switch the adapter at run time (None / "0" = off); used by the evaluation scripts."""
    global ADAPTER_PATH, _adapter
    ADAPTER_PATH, _adapter = (path or "0"), None


def apply_adapter(vecs: np.ndarray) -> np.ndarray:
    a = adapter()
    if a is None:
        return vecs
    W, b = a
    return _norm(vecs + vecs @ W + b)


def embed_texts(texts: list, raw: bool = False) -> Optional[np.ndarray]:
    """["fire", ...] -> (N, D) L2-normalised float32, or None. Cached per text.
    raw=True skips the adapter (what the adapter itself is trained on)."""
    texts = [t.strip() for t in texts]
    if not available() or not texts or not all(texts):
        return None
    todo = [t for t in dict.fromkeys(texts) if t not in _text_cache]
    if todo:
        _load()
        if _model is None:
            return None
        import torch
        with torch.no_grad():
            inputs = _processor(text=todo, return_tensors="pt", padding=True,
                                truncation=True).to(_device)
            feats = _norm(_tensor(_model.get_text_features(**inputs)).detach().cpu().numpy())
        if len(_text_cache) > 4096:
            _text_cache.clear()
        _text_cache.update(zip(todo, feats))
    vecs = np.stack([_text_cache[t] for t in texts])
    return vecs if raw else apply_adapter(vecs)


def embed_text(text: str) -> Optional[np.ndarray]:
    vecs = embed_texts([text])
    return None if vecs is None else vecs[0]


def to_blob(vec: Optional[np.ndarray]) -> Optional[bytes]:
    return None if vec is None else vec.astype(np.float32).tobytes()


def from_blob(blob: Optional[bytes]) -> Optional[np.ndarray]:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32)
