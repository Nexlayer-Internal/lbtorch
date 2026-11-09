#pragma once

// lbtorch extension: Python <-> C++ AdamW checkpoint interoperability.
//
// See docs/adamw_checkpoint_interop.md for the full design rationale and
// the exact byte layout. Short version: `torch::optim::AdamW`'s native
// save()/load() uses `torch::serialize::OutputArchive`/`InputArchive`
// (ultimately a zip of individually-named tensors), which has no defined
// mapping to/from Python's `torch.optim.AdamW.state_dict()` (a nested
// Python dict of Python ints/dicts/tensors, usually itself pickled inside
// `torch.save`'s zip format). Rather than try to byte-match Python's
// pickle-based representation from pure C++ (which would either require
// embedding a Python/pickle-compatible decoder, defeating the point of
// this project, or reverse-engineering pickle's opcode stream), we define
// one explicit, documented exchange format and provide a reader/writer on
// each side: this file for C++, `tools/checkpoint_bridge/
// adamw_checkpoint_bridge.py` for Python. Both are plain struct-packed
// binary I/O with no pickle/zip/jit dependency.
#include <torch/optim/adamw.h>

#include <string>

namespace torch::optim::checkpoint_bridge {

// Writes `optimizer`'s current parameter values + AdamW state (exp_avg,
// exp_avg_sq, max_exp_avg_sq if amsgrad, and step) to `path` in the lbtorch
// AdamW checkpoint exchange format (version 1). Params are flattened across
// all of `optimizer`'s param_groups, in order -- this matches the flat
// global index space Python's `Optimizer.state_dict()["state"]` uses, which
// is what makes a single-param-group round trip (the common case: one
// AdamW instance over one model's parameters, as in the GRPO actor/critic
// training loop this project targets) line up index-for-index with no
// group-boundary bookkeeping needed. Multiple param groups are **not**
// disambiguated by this version of the format -- see the doc for the exact
// limitation and how to extend it.
TORCH_API void export_checkpoint(const AdamW& optimizer, const std::string& path);

// Inverse of `export_checkpoint`. Loads a checkpoint written either by
// `export_checkpoint` or by the companion Python helper
// (`tools/checkpoint_bridge/adamw_checkpoint_bridge.py:export_adamw_checkpoint`)
// into `optimizer`, overwriting its current parameter *values* in place
// (not re-binding the Tensors -- `optimizer`'s param_groups must already
// reference the right parameter tensors, of matching shape/dtype/order) and
// replacing its AdamW state (exp_avg/exp_avg_sq/max_exp_avg_sq/step).
TORCH_API void import_checkpoint(AdamW& optimizer, const std::string& path);

} // namespace torch::optim::checkpoint_bridge
