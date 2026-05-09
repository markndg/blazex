"""
load_model.py — load a blazex pytorch-export directory into a dict of torch tensors.

Usage:
    from load_model import load_tensors
    tensors = load_tensors("./exported/")
    print(tensors["model.embed_tokens.weight"].shape)
"""

import json
from pathlib import Path
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def load_tensors(export_dir: str, as_torch: bool = True) -> dict:
    """Load all tensors from a blazex pytorch export directory.

    Returns:
        dict mapping tensor name → torch.Tensor (or numpy ndarray if as_torch=False)
    """
    export_dir = Path(export_dir)
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found in {export_dir}")

    manifest = json.loads(manifest_path.read_text())
    tensors = {}

    for entry in manifest["tensors"]:
        name = entry["name"]
        fpath = export_dir / entry["file"]
        dtype = np.dtype(entry["numpy_dtype"])
        shape = tuple(entry["shape"])

        raw = np.fromfile(str(fpath), dtype=dtype)
        if shape:
            raw = raw.reshape(shape)

        if as_torch and HAS_TORCH:
            # bfloat16 requires special handling in older numpy
            if entry["numpy_dtype"] == "bfloat16":
                raw = torch.tensor(raw.view(np.uint16)).view(torch.bfloat16).reshape(shape)
            else:
                raw = torch.from_numpy(raw)
        tensors[name] = raw

    return tensors


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    t = load_tensors(path, as_torch=HAS_TORCH)
    print(f"Loaded {len(t)} tensors")
    for name, tensor in list(t.items())[:5]:
        if HAS_TORCH:
            print(f"  {name}: {tensor.shape} {tensor.dtype}")
        else:
            print(f"  {name}: {tensor.shape} {tensor.dtype}")
