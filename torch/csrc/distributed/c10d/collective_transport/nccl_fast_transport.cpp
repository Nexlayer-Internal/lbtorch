#include <torch/csrc/distributed/c10d/collective_transport/nccl_fast_transport.h>

namespace torch::collectives {

void NcclFastTransport::all_gather(const at::Tensor& input_shard, at::Tensor& output) {
  at::Tensor out = output;
  at::Tensor in = input_shard;
  auto work = pg_->all_gather_single(out, in);
  work->wait();
}

void NcclFastTransport::reduce_scatter(const at::Tensor& input, at::Tensor& output_shard) {
  at::Tensor out = output_shard;
  at::Tensor in = input;
  ::c10d::ReduceScatterOptions opts;
  opts.reduceOp = ::c10d::ReduceOp::SUM;
  auto work = pg_->reduce_scatter_single(out, in, opts);
  work->wait();
}

} // namespace torch::collectives
