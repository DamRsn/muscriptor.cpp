"""Capture every intermediate tensor of a reference forward pass.

The port is debugged by bisection: when the final logits disagree, the first
stage whose output disagrees tells you where the bug is. That only works if the
reference numbers come from the *real* implementation, so this module leans on
``register_forward_hook`` -- which hands back both a module's inputs and its
outputs -- rather than on a re-implementation that could silently drift.

Three values live outside any module and are captured by wrapping the function
that produces them: the STFT magnitudes (``torch.stft``), the sinusoidal
position embedding (``create_sin_embedding``) and the masked logits that
argmax actually sees (``LMModel._compute_logits``). Wrapping delegates to the
original in every case; nothing here re-derives a value the model computes.

Layout convention for everything recorded: a ``[1, T, D]`` activation is stored
as a numpy array of shape ``(T, D)``, which GGUF writes as ggml ``ne = [D, T]``
-- i.e. the D components of one timestep are contiguous, the layout the C++
graph uses.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import numpy as np
import torch

import muscriptor.modules.transformer as transformer_module
from muscriptor.models.lm import LMModel


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach to contiguous float32 numpy, dropping a leading batch dim of 1.

    fp16 references are upcast here: the comparison is always done in fp32 so
    that the two references and the ggml output are directly commensurable.
    """
    out = tensor.detach().to(torch.float32).cpu()
    if out.dim() > 1 and out.shape[0] == 1:
        out = out[0]
    return np.ascontiguousarray(out.numpy())


class Tracer:
    """Collects named tensors from one reference run.

    ``detailed`` gates the expensive per-layer capture so it only happens on the
    prefill pass; the decode loop re-enters the same modules thousands of times
    and only its logits are interesting.
    """

    def __init__(self) -> None:
        self.tensors: dict[str, np.ndarray] = {}
        self.detailed = True
        self._logits_calls = 0

    # -- recording ---------------------------------------------------------
    def record(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.tensors:
            raise KeyError(f"tensor {name!r} recorded twice")
        self.tensors[name] = _to_numpy(tensor)

    def _record_detailed(self, name: str, tensor: torch.Tensor) -> None:
        if self.detailed:
            self.record(name, tensor)

    # -- hook factories ----------------------------------------------------
    def _on_output(self, name: str, index: int | None = None):
        def hook(_module, _inputs, output):
            value = output if index is None else output[index]
            self._record_detailed(name, value)

        return hook

    def _on_input(self, name: str):
        def hook(_module, inputs, _output):
            self._record_detailed(name, inputs[0])

        return hook

    # -- installation ------------------------------------------------------
    @contextlib.contextmanager
    def attach(self, model: LMModel, num_layers: int) -> Iterator["Tracer"]:
        handles: list[Any] = []

        def hook_out(module: torch.nn.Module, name: str, index: int | None = None):
            handles.append(module.register_forward_hook(self._on_output(name, index)))

        def hook_in(module: torch.nn.Module, name: str):
            handles.append(module.register_forward_hook(self._on_input(name)))

        conditioners = model.condition_provider.conditioners
        mel_cond = conditioners["self_wav"]

        # Conditioning path (fp32 in the reference even when the transformer is
        # fp16 -- LMModel.forward casts at the torch.cat seam).
        hook_out(mel_cond.mel_spec_transform, "_cond.mel_bchw")
        hook_in(mel_cond.output_proj, "cond.logmel")
        hook_out(mel_cond.output_proj, "cond.proj")
        hook_out(mel_cond, "cond.embed", index=0)
        hook_out(conditioners["instrument_group"].embed, "cond.instrument_group")
        hook_out(conditioners["dataset_name"].embed, "cond.dataset_name")

        # Prefix assembly.
        hook_out(model.emb, "pre.tok_embed")
        hook_in(model.transformer, "pre.prefix")
        hook_in(model.transformer.layers[0], "pre.layer_in")

        # Per-layer outputs: the bisection ladder.
        for i in range(num_layers):
            hook_out(model.transformer.layers[i], f"blk.{i}.out")

        # Layer 0 internals, to localise a failure inside a block.
        layer0 = model.transformer.layers[0]
        hook_out(layer0.norm1, "blk.0.norm1")
        hook_in(layer0.self_attn.out_proj, "blk.0.attn_ctx")
        hook_out(layer0.self_attn, "blk.0.attn_out")
        hook_out(layer0.norm2, "blk.0.norm2")
        hook_out(layer0.linear1, "blk.0.ffn_pre_gelu")
        hook_in(layer0.linear2, "blk.0.ffn_gelu")
        hook_out(layer0.linear2, "blk.0.ffn_out")

        # Output head.
        hook_out(model.out_norm, "post.out_norm")

        original_stft = torch.stft
        original_sin = transformer_module.create_sin_embedding
        original_logits = LMModel._compute_logits

        def traced_stft(*args, **kwargs):
            spec = original_stft(*args, **kwargs)
            # [B, n_freq, T] complex -> (T, n_freq) magnitudes. power=1.0 in the
            # model config, so magnitude is exactly what feeds the mel matmul.
            self._record_detailed("in.spectrum", spec.abs().transpose(-1, -2))
            return spec

        def traced_sin(*args, **kwargs):
            emb = original_sin(*args, **kwargs)
            self._record_detailed("pre.pos_emb", emb)
            return emb

        def traced_logits(lm_self, *args, **kwargs):
            logits = original_logits(lm_self, *args, **kwargs)
            step = self._logits_calls
            self._logits_calls += 1
            # Post-mask, fp32, last timestep only: exactly what argmax sees.
            self.record(f"dec.step{step}.logits", logits)
            # The prefill has now completed, so stop the per-layer capture: the
            # decode loop re-enters every hooked module once per token and would
            # otherwise both blow up memory and collide on names.
            self.detailed = False
            return logits

        torch.stft = traced_stft
        transformer_module.create_sin_embedding = traced_sin
        LMModel._compute_logits = traced_logits
        try:
            yield self
        finally:
            torch.stft = original_stft
            transformer_module.create_sin_embedding = original_sin
            LMModel._compute_logits = original_logits
            for handle in handles:
                handle.remove()

    # -- post-processing ---------------------------------------------------
    def finalize(self, max_decode_steps: int) -> dict[str, np.ndarray]:
        """Normalise layouts and drop what the C++ suite doesn't consume.

        ``dec.step0.logits`` is the prefill's masked logits, so it is also
        published under ``post.logits`` -- the two are the same tensor viewed as
        the end of the prefill ladder and the start of the decode ladder.
        """
        out = dict(self.tensors)

        # mel_spec_transform emits [1, 1, n_mels, T]; the conditioner rearranges
        # to [1, T, n_mels] before the log. Store the post-rearrange layout so
        # it lines up with cond.logmel.
        mel = out.pop("_cond.mel_bchw")
        out["cond.mel"] = np.ascontiguousarray(mel.reshape(mel.shape[-2:]).T)

        if "dec.step0.logits" in out:
            out["post.logits"] = out["dec.step0.logits"]

        for step in list(out):
            if step.startswith("dec.step"):
                index = int(step[len("dec.step") : -len(".logits")])
                if index > max_decode_steps:
                    del out[step]

        # Residual streams the model computes as bare adds. Recorded as derived
        # so a mismatch localises to the residual rather than to the next norm.
        out["blk.0.res1"] = out["pre.layer_in"] + out["blk.0.attn_out"]
        return out
