"""Tooling for the MuScriptor PyTorch -> ggml port.

The port is validated stage by stage: `dump_refs` runs the reference PyTorch
implementation and records every intermediate tensor, `convert_gguf` turns the
released safetensors checkpoint into a GGUF the C++ side can load, and the
Catch2 suite under ``cpp/tests`` replays the same stages in ggml and compares.
"""
