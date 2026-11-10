# Standalone build for the collective_transport library + its tests.
#
# This is deliberately NOT wired into the main libtorch CMakeLists.txt /
# build_variables.bzl: FSDP-DESIGN.md Sec 4 ("Phase 0") specifies this
# transport layer as "an independently buildable/testable C++ library", not
# yet wired into FSDP or anything else, so it is built and validated here on
# its own merits against an already-built libtorch (with
# -DUSE_DISTRIBUTED=1 -DUSE_NCCL=1), without touching the monolithic build.
#
# Usage (see also test/run_multiproc.sh for how the test binaries are
# actually launched):
#   cmake -G Ninja \
#     -DTORCH_SRC_DIR=/workspace \
#     -DTORCH_BUILD_DIR=/workspace/build \
#     -B build .
#   ninja -C build
cmake_minimum_required(VERSION 3.18)
project(lbtorch_collective_transport CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT DEFINED TORCH_SRC_DIR)
  set(TORCH_SRC_DIR "/workspace")
endif()
if(NOT DEFINED TORCH_BUILD_DIR)
  set(TORCH_BUILD_DIR "${TORCH_SRC_DIR}/build")
endif()
if(NOT DEFINED CUDA_HOME)
  set(CUDA_HOME "/usr/local/cuda")
endif()

# Same include-dir set libtorch's own CMake build uses for
# torch/csrc/api/src sources (extracted from this exact checkout's own
# compile_commands.json for torch/csrc/api/src/optim/adamw.cpp), so this
# standalone build sees identical headers/macros to the ones libtorch_cpu
# and libtorch_cuda were themselves compiled with.
set(LBTORCH_INCLUDE_DIRS
  ${TORCH_SRC_DIR}
  ${TORCH_SRC_DIR}/aten/src
  ${TORCH_SRC_DIR}/torch/csrc/api/include
  ${TORCH_SRC_DIR}/torch/csrc
  ${TORCH_SRC_DIR}/torch/csrc/api
  ${TORCH_SRC_DIR}/torch/headeronly
  ${TORCH_SRC_DIR}/c10/..
  ${TORCH_BUILD_DIR}
  ${TORCH_BUILD_DIR}/aten/src
  ${TORCH_BUILD_DIR}/caffe2/aten/src
  ${TORCH_BUILD_DIR}/caffe2/aten/src/TH
  ${TORCH_SRC_DIR}/caffe2/aten/src/TH
  ${TORCH_SRC_DIR}/third_party/fmt/include
  ${CUDA_HOME}/include
)

add_library(lbtorch_collective_transport STATIC
  topology.cpp
  nccl_fast_transport.cpp
  host_staged_transport.cpp
  transport_factory.cpp
)
target_include_directories(lbtorch_collective_transport PUBLIC ${LBTORCH_INCLUDE_DIRS})
target_compile_definitions(lbtorch_collective_transport PUBLIC
  USE_DISTRIBUTED
  USE_C10D_NCCL
  AT_PER_OPERATOR_HEADERS
)
target_link_directories(lbtorch_collective_transport PUBLIC ${TORCH_BUILD_DIR}/lib ${CUDA_HOME}/lib64)
target_link_libraries(lbtorch_collective_transport PUBLIC
  torch_cpu torch_cuda c10 c10_cuda
  cudart nccl
  pthread rt
)

function(lbtorch_ct_test name)
  add_executable(${name} test/${name}.cpp)
  target_link_libraries(${name} PRIVATE lbtorch_collective_transport)
endfunction()

lbtorch_ct_test(test_collective_correctness)
lbtorch_ct_test(test_bounded_wait)
lbtorch_ct_test(repro_vanilla_nccl_hang)
lbtorch_ct_test(repro_transport_no_hang)
