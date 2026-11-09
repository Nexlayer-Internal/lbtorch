"""Python-side implementation of the lbtorch AdamW checkpoint exchange
format (version 1), the counterpart to
`torch/csrc/api/src/optim/adamw_checkpoint_bridge.cpp`.

Why this exists / why it's a flat custom binary format instead of trying to
read/write `torch.optim.AdamW.state_dict()`'s own representation directly:
`state_dict()` is a nested Python dict (`{'state': {idx: {...}},
'param_groups': [...]}`) that in practice always gets serialized via
`torch.save`, i.e. pickle inside a zip container. Reproducing that exact
byte-for-byte representation from pure C++ (with no embedded Python/pickle
interpreter -- the entire point of lbtorch is to not need one) is
impractical. Instead, this module and its C++ counterpart agree on one
simple, explicit, documented binary layout and each side converts to/from
their own native representation (`state_dict()` here, `AdamWParamState`
there). See docs/adamw_checkpoint_interop.md for the full rationale and the
byte layout spec; this file's `_MAGIC`/struct-pack calls ARE that spec, kept
in sync with the C++ side's constants.

No pickle, no `torch.save`/`torch.load`, no jit archive -- plain binary I/O
via `struct` + `numpy`, so there is nothing in this file that depends on a
matching pickle protocol version or a matching PyTorch version between the
process that wrote the checkpoint and the one that reads it.
"""
from __future__ import annotations

import struct
from typing import Optional

import numpy as np
import torch

_MAGIC = b"LBA1"
_VERSION = 1

_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float64: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}
_DTYPE_TO_NUMPY_STORAGE = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    # bfloat16 has no native numpy dtype; stored bit-for-bit as int16.
    torch.bfloat16: np.int16,
}


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().to("cpu").contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def _bytes_to_tensor(data: bytes, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    np_dtype = _DTYPE_TO_NUMPY_STORAGE[dtype]
    arr = np.frombuffer(data, dtype=np_dtype).reshape(shape).copy()
    t = torch.from_numpy(arr)
    if dtype == torch.bfloat16:
        t = t.view(torch.bfloat16)
    return t


def export_adamw_checkpoint(optimizer: torch.optim.AdamW, path: str) -> None:
    """Writes `optimizer`'s current parameter values + AdamW state to `path`
    in the lbtorch AdamW checkpoint exchange format.

    Params are flattened across all of `optimizer.param_groups`, in order,
    matching the flat global index space `Optimizer.state_dict()["state"]`
    already uses internally -- see the C++ header's docstring for why that's
    what makes this line up with a C++-side single-param-group optimizer
    with no extra bookkeeping.
    """
    sd = optimizer.state_dict()
    state = sd["state"]
    # Flat, in-order param list (Optimizer.state_dict() already numbers
    # params 0..N-1 in exactly this order across all param_groups).
    params = [p for group in optimizer.param_groups for p in group["params"]]

    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<i", _VERSION))
        f.write(struct.pack("<q", len(params)))

        for idx, p in enumerate(params):
            st = state.get(idx)
            if st is None:
                raise RuntimeError(
                    f"adamw_checkpoint_bridge: param {idx} has no optimizer "
                    "state yet -- call optimizer.step() at least once before "
                    "exporting a checkpoint"
                )
            shape = tuple(p.shape)
            dtype = p.dtype
            has_max = "max_exp_avg_sq" in st and st["max_exp_avg_sq"] is not None

            f.write(struct.pack("<i", len(shape)))
            for s in shape:
                f.write(struct.pack("<q", s))
            f.write(struct.pack("<i", _DTYPE_TO_CODE[dtype]))
            f.write(struct.pack("<B", 1 if has_max else 0))

            step = st["step"]
            step_value = float(step.item()) if torch.is_tensor(step) else float(step)
            f.write(struct.pack("<d", step_value))

            f.write(_tensor_to_bytes(p.data.to(dtype)))
            f.write(_tensor_to_bytes(st["exp_avg"].to(dtype)))
            f.write(_tensor_to_bytes(st["exp_avg_sq"].to(dtype)))
            if has_max:
                f.write(_tensor_to_bytes(st["max_exp_avg_sq"].to(dtype)))


def import_adamw_checkpoint(optimizer: torch.optim.AdamW, path: str) -> None:
    """Inverse of `export_adamw_checkpoint`: loads a checkpoint written
    either by it or by the C++ `adamw_checkpoint_bridge::export_checkpoint`
    into `optimizer` in place (param values + AdamW state), via the normal
    `optimizer.load_state_dict()` path so every other piece of PyTorch that
    expects a real state dict (LR schedulers, etc.) keeps working.
    """
    params = [p for group in optimizer.param_groups for p in group["params"]]

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != _MAGIC:
            raise ValueError(f"'{path}' is not an lbtorch AdamW checkpoint (bad magic)")
        (version,) = struct.unpack("<i", f.read(4))
        if version != _VERSION:
            raise ValueError(f"unsupported lbtorch AdamW checkpoint version {version}")
        (num_params,) = struct.unpack("<q", f.read(8))
        if num_params != len(params):
            raise ValueError(
                f"checkpoint has {num_params} params but optimizer has "
                f"{len(params)} -- construct the optimizer over the matching "
                "model params before importing"
            )

        new_state = {}
        for idx in range(num_params):
            (ndim,) = struct.unpack("<i", f.read(4))
            shape = struct.unpack(f"<{ndim}q", f.read(8 * ndim))
            (dtype_code,) = struct.unpack("<i", f.read(4))
            dtype = _CODE_TO_DTYPE[dtype_code]
            (has_max,) = struct.unpack("<B", f.read(1))
            (step_value,) = struct.unpack("<d", f.read(8))

            p = params[idx]
            if tuple(p.shape) != shape:
                raise ValueError(
                    f"param {idx} shape mismatch: checkpoint has {shape}, "
                    f"optimizer param has {tuple(p.shape)}"
                )

            def read_tensor() -> torch.Tensor:
                n = 1
                for s in shape:
                    n *= s
                itemsize = np.dtype(_DTYPE_TO_NUMPY_STORAGE[dtype]).itemsize
                data = f.read(n * itemsize)
                return _bytes_to_tensor(data, shape, dtype)

            param_data = read_tensor()
            exp_avg = read_tensor()
            exp_avg_sq = read_tensor()
            max_exp_avg_sq: Optional[torch.Tensor] = read_tensor() if has_max else None

            with torch.no_grad():
                p.data.copy_(param_data.to(p.dtype).to(p.device))

            entry = {
                "step": torch.tensor(step_value, dtype=torch.float32),
                "exp_avg": exp_avg.to(p.dtype).to(p.device),
                "exp_avg_sq": exp_avg_sq.to(p.dtype).to(p.device),
            }
            if max_exp_avg_sq is not None:
                entry["max_exp_avg_sq"] = max_exp_avg_sq.to(p.dtype).to(p.device)
            new_state[idx] = entry

    sd = optimizer.state_dict()
    sd["state"] = new_state
    optimizer.load_state_dict(sd)
