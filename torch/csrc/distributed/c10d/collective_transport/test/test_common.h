#pragma once

// Shared bootstrap helper for the collective_transport multi-process test
// binaries: every test here is launched once per rank (see run_multiproc.sh,
// a plain bash process launcher -- no Python anywhere in this build or test
// path, per the project's zero-Python goal) with RANK / WORLD_SIZE /
// MASTER_PORT in the environment, and builds a TCPStore (already pure C++,
// already libtorch) for rendezvous.

#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace torch::collectives::test {

struct ProcEnv {
  int rank;
  int world_size;
  uint16_t port;
};

inline ProcEnv read_proc_env() {
  const char* rank_s = std::getenv("RANK");
  const char* world_size_s = std::getenv("WORLD_SIZE");
  const char* port_s = std::getenv("MASTER_PORT");
  if (!rank_s || !world_size_s || !port_s) {
    throw std::runtime_error("RANK, WORLD_SIZE, and MASTER_PORT must be set in the environment");
  }
  ProcEnv env{};
  env.rank = std::atoi(rank_s);
  env.world_size = std::atoi(world_size_s);
  env.port = static_cast<uint16_t>(std::atoi(port_s));
  return env;
}

inline c10::intrusive_ptr<::c10d::TCPStore> make_tcp_store(const ProcEnv& env) {
  ::c10d::TCPStoreOptions opts;
  opts.port = env.port;
  opts.isServer = (env.rank == 0);
  opts.waitWorkers = false; // each test does its own explicit rendezvous barriers
  opts.timeout = std::chrono::milliseconds(60000);
  opts.useLibUV = false; // this build has no libuv; fall back to the plain-socket backend
  return c10::make_intrusive<::c10d::TCPStore>("127.0.0.1", opts);
}

} // namespace torch::collectives::test
