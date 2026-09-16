"""Convert a released MuScriptor safetensors checkpoint to GGUF.

Precision policy mirrors what the reference does at load time. ``load_model``
casts the whole model to fp16 and then calls ``condition_provider.float()``,
because log-mel of a quiet passage underflows in half precision. So here:

* **F16** -- token embedding, LM head, and every transformer matmul weight.
* **F32** -- every LayerNorm weight/bias (tiny, and normalisation is where
  precision actually costs you), plus the entire conditioning path: mel
  filterbank, STFT window, mel projection, and the two class embeddings.

Tensor names follow the llama.cpp ``blk.<i>.<part>`` convention, and every
weight is stored in the ``(out_features, in_features)`` numpy layout that torch
already uses -- which GGUF turns into ggml ``ne = [in, out]``, the first
argument ``ggml_mul_mat`` expects. The one exception is the mel filterbank,
which torch stores transposed relative to a Linear; it is transposed here so
every matmul in the C++ graph reads the same way.

Needs the ``convert`` dependency group, and a HuggingFace account with the
upstream licence accepted -- the source checkpoints are gated.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from msl import notice, paths, tensor_io
from msl.checksum import sha256

ARCH = "muscriptor"

# The loader contract: which metadata keys and tensors a file carries and how
# they are read. The C++ loader accepts this value and no other. Bump it when a
# file stops being readable by the previous loader, not when weights change.
FORMAT_VERSION = 1

# Upstream checkpoints, and where this conversion is published.
SOURCE_ORG = "MuScriptor"
GGUF_REPO = "DamRsn/muscriptor-gguf"
CODE_REPO = "https://github.com/DamRsn/muscriptor.cpp"
PAPER_DOI = "10.48550/arXiv.2607.08168"
LICENSE = "cc-by-nc-4.0"
LICENSE_NAME = "Creative Commons Attribution-NonCommercial 4.0 International"
LICENSE_LINK = "https://creativecommons.org/licenses/by-nc/4.0/legalcode.en"
TAGS = [
    "gguf",
    "ggml",
    "music-transcription",
    "audio-to-midi",
    "automatic-music-transcription",
    "muscriptor",
]

# Audio front-end constants. Hard-coded in the reference's `_build_model`
# rather than stored in the checkpoint, so they are carried into the GGUF
# explicitly instead of being re-hard-coded in C++.
SAMPLE_RATE = 16_000
N_FFT = 2048
FRAME_RATE = 100
HOP_LENGTH = SAMPLE_RATE // FRAME_RATE
N_MELS = 512
LOG_EPS = 1e-6
LAYER_NORM_EPS = 1e-5
MAX_PERIOD = 10_000.0
# LMModel._compute_logits forces logits[:, 1393:] to -inf regardless of `card`,
# so `medium`/`large` (card 1395) have two unreachable ids that `small` lacks.
LOGIT_MASK_START = 1393

_COND = "condition_provider.conditioners"
_MEL = f"{_COND}.self_wav.mel_spec_transform"

# source key -> (gguf name, keep in fp32, transform)
_GLOBAL_MAP: dict[str, tuple[str, bool, Callable[[np.ndarray], np.ndarray] | None]] = {
    "emb.0.weight": ("token_embd.weight", False, None),
    "linears.0.weight": ("output.weight", False, None),
    "out_norm.weight": ("output_norm.weight", True, None),
    "out_norm.bias": ("output_norm.bias", True, None),
    f"{_MEL}.mel_scale.fb": ("cond.mel_fb.weight", True, np.transpose),
    f"{_MEL}.spectrogram.window": ("cond.stft_window", True, None),
    f"{_COND}.self_wav.output_proj.weight": ("cond.proj.weight", True, None),
    f"{_COND}.self_wav.output_proj.bias": ("cond.proj.bias", True, None),
    f"{_COND}.instrument_group.embed.weight": ("cond.instrument_group.weight", True, None),
    f"{_COND}.dataset_name.embed.weight": ("cond.dataset_name.weight", True, None),
}

# per-layer suffix -> (gguf suffix, keep in fp32)
_LAYER_MAP: dict[str, tuple[str, bool]] = {
    "norm1.weight": ("attn_norm.weight", True),
    "norm1.bias": ("attn_norm.bias", True),
    "norm2.weight": ("ffn_norm.weight", True),
    "norm2.bias": ("ffn_norm.bias", True),
    "self_attn.in_proj_weight": ("attn_qkv.weight", False),
    "self_attn.out_proj.weight": ("attn_out.weight", False),
    "linear1.weight": ("ffn_up.weight", False),
    "linear2.weight": ("ffn_down.weight", False),
}

_LAYER_RE = re.compile(r"^transformer\.layers\.(\d+)\.(.+)$")

# What the reference reads out of a published repo's config.json.
_CONFIG_FIELDS = ("dim", "num_heads", "num_layers", "card")


def _translate(
    state_dict: dict[str, np.ndarray],
    num_layers: int,
    weight_dtype: str = "f16",
) -> dict[str, np.ndarray]:
    """Apply the name map and precision policy. Raises on anything unmapped.

    ``weight_dtype="f32"`` overrides the policy and keeps *everything* in fp32.
    That build is not for shipping -- it exists as a diagnostic. If a stage
    disagrees with the reference under F16 weights but matches under F32, the
    graph is right and the gap is precision; if it disagrees under both, the
    graph is wrong. Without that split every mismatch is ambiguous.
    """
    if weight_dtype not in ("f16", "f32"):
        raise ValueError(f"weight_dtype must be 'f16' or 'f32', got {weight_dtype!r}")
    force_f32 = weight_dtype == "f32"

    out: dict[str, np.ndarray] = {}
    seen: set[str] = set()

    for key, value in state_dict.items():
        match = _LAYER_RE.match(key)
        if match:
            index, suffix = int(match.group(1)), match.group(2)
            if suffix not in _LAYER_MAP:
                raise KeyError(f"unmapped per-layer tensor: {key}")
            name_suffix, keep_f32 = _LAYER_MAP[suffix]
            name, transform = f"blk.{index}.{name_suffix}", None
        elif key in _GLOBAL_MAP:
            name, keep_f32, transform = _GLOBAL_MAP[key]
        else:
            raise KeyError(f"unmapped tensor: {key}")

        array = value if transform is None else transform(value)
        out[name] = np.ascontiguousarray(
            array.astype(np.float32 if (keep_f32 or force_f32) else np.float16)
        )
        seen.add(key)

    missing = set(_GLOBAL_MAP) - seen
    if missing:
        raise KeyError(f"checkpoint is missing expected tensors: {sorted(missing)}")
    expected = num_layers * len(_LAYER_MAP) + len(_GLOBAL_MAP)
    if len(out) != expected:
        raise ValueError(f"produced {len(out)} tensors, expected {expected}")
    return out


def _size_label(params: int) -> str:
    """HuggingFace-style parameter count: ``103M``, ``1.4B``."""
    return f"{params / 1e9:.1f}B" if params >= 1e9 else f"{params / 1e6:.0f}M"


def _download(size: str) -> tuple[Path, dict[str, int], str]:
    """@return The cached weights, the parsed config, and the source revision."""
    from huggingface_hub import hf_hub_download

    repo_id = f"{SOURCE_ORG}/muscriptor-{size}"
    weights = Path(hf_hub_download(repo_id, "model.safetensors"))
    # hf_hub_download returns <cache>/snapshots/<revision>/<filename>. Pinning
    # the second download to that revision keeps the pair consistent even if
    # the branch moves between the two calls.
    revision = weights.parent.name
    config_path = Path(hf_hub_download(repo_id, "config.json", revision=revision))
    config = json.loads(config_path.read_text())
    return weights, {field: config[field] for field in _CONFIG_FIELDS}, revision


def _metadata(
    size: str,
    config: dict[str, int],
    tensors: dict[str, np.ndarray],
    revision: str,
    source_sha256: str,
    weight_dtype: str,
) -> dict[str, Any]:
    """Every KV pair the file carries, beyond ``general.architecture``."""
    import gguf

    name = f"MuScriptor {size.capitalize()}"
    source_url = f"https://huggingface.co/{SOURCE_ORG}/muscriptor-{size}"
    file_type = (
        gguf.LlamaFileType.ALL_F32 if weight_dtype == "f32" else gguf.LlamaFileType.MOSTLY_F16
    )
    return {
        "general.type": "model",
        "general.quantization_version": gguf.GGML_QUANT_VERSION,
        "general.file_type": int(file_type),
        "general.name": name,
        "general.basename": "muscriptor",
        "general.size_label": _size_label(sum(t.size for t in tensors.values())),
        "general.author": "Mirelo and Kyutai",
        "general.organization": SOURCE_ORG,
        "general.quantized_by": GGUF_REPO.split("/")[0],
        "general.description": (
            "Format conversion of MuScriptor to GGUF for muscriptor.cpp. "
            + (
                "All weights kept in float32. "
                if weight_dtype == "f32"
                else "Bulk weights cast to float16; conditioning path and normalisation "
                "kept in float32. "
            )
            + "Not retrained or fine-tuned."
        ),
        "general.license": LICENSE,
        "general.license.name": LICENSE_NAME,
        "general.license.link": LICENSE_LINK,
        "general.url": f"https://huggingface.co/{GGUF_REPO}",
        "general.repo_url": CODE_REPO,
        "general.source.url": source_url,
        "general.source.repo_url": "https://github.com/muscriptor/muscriptor",
        "general.base_model.count": 1,
        "general.base_model.0.name": name,
        "general.base_model.0.organization": SOURCE_ORG,
        "general.base_model.0.author": "Mirelo and Kyutai",
        "general.base_model.0.repo_url": source_url,
        "general.base_model.0.version": revision,
        "general.base_model.0.doi": PAPER_DOI,
        "general.tags": TAGS,
        f"{ARCH}.format_version": FORMAT_VERSION,
        f"{ARCH}.license.notice": notice.LICENSE_NOTICE,
        f"{ARCH}.embedding_length": config["dim"],
        f"{ARCH}.block_count": config["num_layers"],
        f"{ARCH}.attention.head_count": config["num_heads"],
        f"{ARCH}.attention.head_dim": config["dim"] // config["num_heads"],
        f"{ARCH}.feed_forward_length": tensors["blk.0.ffn_up.weight"].shape[0],
        f"{ARCH}.vocab_size": config["card"],
        f"{ARCH}.initial_token_id": config["card"],
        f"{ARCH}.logit_mask_start": LOGIT_MASK_START,
        f"{ARCH}.attention.layer_norm_epsilon": LAYER_NORM_EPS,
        f"{ARCH}.position_embedding.max_period": MAX_PERIOD,
        f"{ARCH}.audio.sample_rate": SAMPLE_RATE,
        f"{ARCH}.audio.n_fft": N_FFT,
        f"{ARCH}.audio.hop_length": HOP_LENGTH,
        f"{ARCH}.audio.frame_rate": FRAME_RATE,
        f"{ARCH}.audio.n_mels": N_MELS,
        f"{ARCH}.audio.log_eps": LOG_EPS,
        # Pins the file to the exact upstream bytes it was made from.
        f"{ARCH}.source.revision": revision,
        f"{ARCH}.source.sha256": source_sha256,
        f"{ARCH}.source.weight_dtype": weight_dtype,
    }


def convert(size: str, output: Path | None = None, weight_dtype: str = "f16") -> Path:
    """Download (or reuse the cache for) ``size`` and write the GGUF."""
    from safetensors.numpy import load_file

    paths.ensure_dirs()
    weights_path, config, revision = _download(size)
    tensors = _translate(load_file(str(weights_path)), config["num_layers"], weight_dtype)

    output = output or (paths.WEIGHTS_DIR / f"muscriptor-{size}-{weight_dtype}.gguf")
    metadata = _metadata(size, config, tensors, revision, sha256(weights_path), weight_dtype)
    tensor_io.write_gguf(output, ARCH, tensors, metadata)
    return output


def validate(size: str, gguf_path: Path, weight_dtype: str = "f16") -> None:
    """Round-trip check: every GGUF tensor must equal its safetensors source.

    Catches a wrong name map, a missed transpose, or a tensor silently written
    with the wrong shape -- all of which would otherwise surface much later as
    an inexplicable numerical mismatch in the C++ suite.
    """
    from safetensors.numpy import load_file

    weights_path, config, _ = _download(size)
    expected = _translate(load_file(str(weights_path)), config["num_layers"], weight_dtype)
    actual = tensor_io.read_gguf(gguf_path)

    if set(expected) != set(actual):
        raise AssertionError(
            f"tensor name mismatch: only-in-source={sorted(set(expected) - set(actual))}, "
            f"only-in-gguf={sorted(set(actual) - set(expected))}"
        )
    for name, want in expected.items():
        got = actual[name]
        if got.shape != want.shape:
            raise AssertionError(f"{name}: shape {got.shape} != {want.shape}")
        if got.dtype != want.dtype:
            raise AssertionError(f"{name}: dtype {got.dtype} != {want.dtype}")
        if not np.array_equal(got, want):
            raise AssertionError(f"{name}: values differ after round-trip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="medium", choices=("small", "medium", "large"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--weight-dtype",
        default="f16",
        choices=("f16", "f32"),
        help="f16 is the shipping policy; f32 is a diagnostic build that keeps "
        "every tensor in fp32, to separate precision loss from graph bugs",
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    out = convert(args.size, args.output, args.weight_dtype)
    size_mb = out.stat().st_size / (1 << 20)
    print(f"wrote {out} ({size_mb:.1f} MiB)")
    if not args.no_validate:
        validate(args.size, out, args.weight_dtype)
        print("round-trip validation passed")
    print(f"sha256  {sha256(out)}  {out.name}")
