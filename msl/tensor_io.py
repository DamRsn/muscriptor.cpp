"""GGUF read/write helpers shared by the weight converter and the ref dumper.

Both artifacts are GGUF for the same reason: ggml reads it natively, so the C++
test suite needs no bespoke parser and no float round-tripping through text.

Shape convention: gguf-py stores a
numpy array of shape ``(n1, n0)`` as ggml ``ne = [n0, n1]``. So the last numpy
axis is the contiguous ggml axis, and a torch ``nn.Linear`` weight -- numpy
``(out_features, in_features)`` -- lands as ggml ``[in, out]``, exactly what
``ggml_mul_mat`` wants as its first argument.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gguf
import numpy as np


# Spec-typed as uint32. Every other integer key here is ours, and the C++
# loader reads them as int32.
_UINT32_KEYS = (
    "general.file_type",
    "general.quantization_version",
    "general.base_model.count",
)


def _add_metadata(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    if isinstance(value, bool):
        writer.add_bool(key, value)
    elif isinstance(value, list):
        writer.add_array(key, value)
    elif key in _UINT32_KEYS:
        writer.add_uint32(key, value)
    elif isinstance(value, int):
        writer.add_int32(key, value)
    elif isinstance(value, float):
        writer.add_float32(key, value)
    else:
        writer.add_string(key, str(value))


def write_gguf(
    path: Path,
    arch: str,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write ``tensors`` (and scalar ``metadata``) to a GGUF file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(path), arch)
    for key, value in (metadata or {}).items():
        _add_metadata(writer, key, value)
    for name, array in tensors.items():
        if array.dtype not in (np.float32, np.float16):
            raise TypeError(
                f"{name}: GGUF payload must be float32 or float16, got {array.dtype}. "
                "Integer data (token ids, argmax) belongs in the JSON manifest."
            )
        writer.add_tensor(name, np.ascontiguousarray(array))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def read_gguf(path: Path) -> dict[str, np.ndarray]:
    """Read a GGUF back into ``{name: numpy array}`` with the write-side shape."""
    reader = gguf.GGUFReader(str(path))
    # gguf-py already presents .data in the numpy shape it was written with
    # (reader.shape is the ggml ne view of the same thing).
    return {tensor.name: tensor.data for tensor in reader.tensors}
