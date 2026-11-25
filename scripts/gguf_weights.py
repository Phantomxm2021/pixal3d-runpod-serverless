"""Load the released Q8_0 tensors into Pixal3D's native PyTorch model."""

from __future__ import annotations

from pathlib import Path


def load_gguf_weights(model, path: str | Path) -> None:
    import gguf
    import numpy as np
    import torch

    reader = gguf.GGUFReader(str(path))
    state = model.state_dict()
    seen = set()
    supported = (
        gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.Q8_0,
    )
    if not any(tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0 for tensor in reader.tensors):
        raise ValueError(f"Checkpoint does not contain Q8_0 weights: {path}")
    with torch.no_grad():
        for tensor in reader.tensors:
            name = tensor.name
            if name in seen or name not in state or tensor.tensor_type not in supported:
                raise ValueError(f"Unexpected GGUF tensor: {name}")
            seen.add(name)
            target = state[name]
            shape_field = reader.get_field(f"comfy.gguf.orig_shape.{name}")
            shape = (
                tuple(int(shape_field.parts[index][0]) for index in shape_field.data)
                if shape_field is not None else tuple(int(size) for size in reversed(tensor.shape))
            )
            if shape != tuple(target.shape):
                raise ValueError(f"Invalid GGUF tensor shape for {name}: {shape} != {tuple(target.shape)}")
            if name == "rope_phases" and target.is_complex():
                # GGUF stores a real version. The constructor recreates the exact complex RoPE table.
                continue
            array = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
            if not array.flags.writeable:
                array = np.array(array)
            target.copy_(torch.from_numpy(array.reshape(shape)))
    missing = state.keys() - seen
    if missing:
        raise ValueError(f"Missing GGUF tensors: {sorted(missing)[:5]}")
