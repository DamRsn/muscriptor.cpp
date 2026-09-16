# Model

[MuScriptor](https://github.com/muscriptor/muscriptor) is an automatic music
transcription model by Kyutai and Mirelo
([paper](https://arxiv.org/abs/2607.08168v1),
[checkpoints](https://huggingface.co/MuScriptor)). It transcribes full mixes
with several instruments at once, and can be told which instruments to
transcribe. The weights are CC BY-NC 4.0.

## Sizes

| Size | Parameters | Layers | Dim | Heads |
|---|---|---|---|---|
| `small` | ~103M | 14 | 768 | 12 |
| `medium` | ~307M | 24 | 1024 | 16 |
| `large` | ~1.4B | 48 | 1536 | 24 |

`msl::Hparams` reads every architecture and front-end constant from the GGUF
metadata, so one build loads any of the three.

## Inputs

16 kHz mono float32, in 5 s chunks of 80,000 samples. Chunks are decoded one
after another, and prelude forcing carries still-sounding notes from one chunk
into the next.

## Outputs

The model outputs token ids from a fixed 1393-token MT3 vocabulary, one
sequence per chunk, each ending in EOS. Integer logic turns these into notes
([`TOKENIZER.md`](TOKENIZER.md)). Each note has:

- onset and offset, on a 10 ms grid
- MIDI pitch
- instrument: a decoded program, or a drum hit

The model does not produce:

- velocity or dynamics
- tempo, beats, meter, key, pitch bend or per-note confidence
- two notes at the same pitch on the same instrument at the same time

## Pipeline

```
16 kHz mono float32 (whole signal)
  │  split into 5 s chunks; the last is zero-padded, and the padding is not masked
  ▼
per chunk:
  │  STFT: reflect padding, periodic Hann(2048), hop 160, magnitudes   → [501, 1025]
  │  mel filterbank (512 HTK bins, stored in the checkpoint)            → [501, 512]
  │  log(mel + 1e-6), projection to dim, mask frames past the audio     → [501, dim]
  │  prefix: [mel, dataset_name, instrument_group…, tokens] + sinusoidal positions
  │  N × pre-norm causal transformer block, KV-cached
  │  out_norm → LM head → ids ≥ 1393 set to −inf                        → logits
  │  optional forbidden-token mask; greedy argmax; repeat until EOS or 2000 tokens,
  │    forced prompt included
  ▼
token ids
  │  MT3 decode state machine → note starts, ends and drum hits
  │  prelude forcing: open notes become the next chunk's forced prompt
  ▼
note assembly: validate, trim overlaps, sort
  ▼
std::vector<msl::Note>
```

The reference runs its fp16 model with the conditioning path (STFT to
projection) kept in fp32; the cast to fp16 happens where the prefix is
assembled. The port keeps those tensors in F32 and always runs the
conditioning front-end on the CPU backend, because ggml's Metal matmul rounds
F32 inputs to half precision.

## Architecture details

A decoder-only pre-norm transformer: multi-head attention and a GELU
feed-forward, with no RoPE, no GQA and no biases on the large matmuls. These
details are easy to get wrong, and each has a dedicated test:

- **Prefix order** is `[mel, dataset_name, instrument_group…, tokens]`, the
  reverse of `ConditioningProvider`'s iteration order.
- **Class embeddings** use row 1 for the null class. Group id `g` is row
  `g + 2`.
- **Sinusoidal positions** put the cosine half first, and use an exponent
  denominator of `half_dim - 1`.
- **GELU** is the exact erf form. `ggml_gelu` is the tanh approximation; the
  port uses `ggml_gelu_erf`.
- **The causal mask** is aligned to the bottom-right, so it also covers a
  prompt that is prefilled in one pass.
- **The last mel frame is always masked.** The mask length comes from the
  waveform (80000 / 160 = 500 frames), while the centred STFT produces 501.
- **The mel filterbank and the STFT window** are read from the checkpoint. The
  stored filterbank differs from MuScriptor's own `melscale_fbanks` by about
  2e-4 relative. The stored window is a periodic Hann that has been through
  fp16.
- **Reflect padding** leaves out the edge sample at both ends: the left pad is
  `x[1024] … x[1]`.
- **The STFT gives magnitudes** (`power = 1.0`), not powers.
- **pffft's real transform** puts DC and Nyquist in the first complex slot:
  `out[0]` is `Re F(0)` and `out[1]` is `Re F(N/2)`.
- **`logits[1393:]` is always −inf**, so `medium` and `large` (vocabulary size
  1395) have two ids that can never be produced.

### KV cache

For each layer, K is `[dim, n_ctx]`: one row per position, heads side by side.
V is stored transposed, `[n_ctx, head_dim, n_head]`, so reading it is a plain
view. Both are F32. `Model` defaults to `n_ctx = 2560`, and `Transcriber`
allocates 2538. On ggml-vulkan, a cache
laid out with one plane per head reads back as zeros, so the layout is fixed.

## Tensor map

Weights keep torch's `(out, in)` layout, which GGUF stores as
`ne = [in, out]`. The mel filterbank is transposed on conversion.

| safetensors | GGUF | Type |
|---|---|---|
| `emb.0.weight` | `token_embd.weight` | F16 |
| `linears.0.weight` | `output.weight` | F16 |
| `out_norm.{weight,bias}` | `output_norm.{weight,bias}` | F32 |
| `transformer.layers.N.norm1.{weight,bias}` | `blk.N.attn_norm.{weight,bias}` | F32 |
| `transformer.layers.N.self_attn.in_proj_weight` | `blk.N.attn_qkv.weight` | F16 |
| `transformer.layers.N.self_attn.out_proj.weight` | `blk.N.attn_out.weight` | F16 |
| `transformer.layers.N.norm2.{weight,bias}` | `blk.N.ffn_norm.{weight,bias}` | F32 |
| `transformer.layers.N.linear1.weight` | `blk.N.ffn_up.weight` | F16 |
| `transformer.layers.N.linear2.weight` | `blk.N.ffn_down.weight` | F16 |
| `…self_wav.mel_spec_transform.mel_scale.fb` | `cond.mel_fb.weight` (transposed) | F32 |
| `…self_wav.mel_spec_transform.spectrogram.window` | `cond.stft_window` | F32 |
| `…self_wav.output_proj.{weight,bias}` | `cond.proj.{weight,bias}` | F32 |
| `…instrument_group.embed.weight` | `cond.instrument_group.weight` | F32 |
| `…dataset_name.embed.weight` | `cond.dataset_name.weight` | F32 |

`in_proj_weight` packs q, k and v with q outermost, which is already ggml's
layout, so q, k and v are views into it with no copy.

With `--weight-dtype f32`, every tensor is F32.

## GGUF metadata

- `muscriptor.format_version`, checked before anything else is read
  ([`API.md`](API.md#checkpoint-format-version)).
- Geometry: `embedding_length`, `block_count`, `attention.head_count`,
  `attention.head_dim`, `feed_forward_length`, `vocab_size`,
  `initial_token_id`, `logit_mask_start`, `attention.layer_norm_epsilon`,
  `position_embedding.max_period`.
- Framing: the six `audio.*` constants.
- Provenance: `source.revision`, `source.sha256`, `source.weight_dtype`.
- `license.notice`: the upstream terms of use, verbatim.
- `general.*`: name, size label, licence, source URLs, base-model record and
  tags, as read by Hugging Face and the llama.cpp tooling.

`msl-convert` checks the written file against the safetensors source and
prints its sha256. `msl-fetch` verifies downloads against the published
`SHA256SUMS`.
