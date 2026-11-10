# Standalone build for the FSDP orchestration layer + its regression test,
# same pattern as ../collective_transport/CMakeLists.txt (this layer's own
# Phase 0 dependency): built and validated against an already-built libtorch
# (-DUSE_DISTRIBUTED=1 -DUSE_NCCL=1), not wired into the monolithic
# torch_cpu/torch_cuda build. FSDP-DESIGN.md Sec 4, "Phase 1-3".
#
# Usage:
#   cmake -G Ninja \
#     -DTORCH_SRC_DIR=/workspace \
#     -DTORCH_BUILD_DIR=/workspace/build \
#     -B build .
#   ninja -C build
cmake_minimum_required(VERSION 3.18)
project(lbtorch_fsdp CXX)

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

# collective_transport is this layer's own Phase 0 dependency -- build it as
# a subdirectory of this same CMake invocation rather than requiring it be
# pre-built separately (its own CMakeLists.txt is unchanged / untouched).
add_subdirectory(
  "${TORCH_SRC_DIR}/torch/csrc/distributed/c10d/collective_transport"
  "${CMAKE_BINARY_DIR}/collective_transport"
)

# Same include-dir set collective_transport's own CMakeLists.txt uses
# (extracted the same way, from this checkout's compile_commands.json).
set(LBTORCH_FSDP_INCLUDE_DIRS
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

add_library(lbtorch_fsdp STATIC
  fsdp_unit.cpp
)
target_include_directories(lbtorch_fsdp PUBLIC ${LBTORCH_FSDP_INCLUDE_DIRS})
target_compile_definitions(lbtorch_fsdp PUBLIC
  USE_DISTRIBUTED
  USE_C10D_NCCL
  AT_PER_OPERATOR_HEADERS
)
target_link_directories(lbtorch_fsdp PUBLIC ${TORCH_BUILD_DIR}/lib ${CUDA_HOME}/lib64)
target_link_libraries(lbtorch_fsdp PUBLIC
  lbtorch_collective_transport
  torch_cpu torch_cuda c10 c10_cuda
  cudart nccl
  pthread rt
)

function(lbtorch_fsdp_test name)
  add_executable(${name} test/${name}.cpp)
  target_link_libraries(${name} PRIVATE lbtorch_fsdp)
  # torch_cuda's CUDA kernel/dispatch-key registrations happen via static
  # initializers in libtorch_cuda.so; the linker's --as-needed default drops
  # that .so from any binary (like test_baseline_single_gpu) that never
  # directly references one of its C++ symbols, even though every CUDA
  # tensor op needs its registrations to have run. Force it to stay linked.
  target_link_options(${name} PRIVATE -Wl,--no-as-needed,-ltorch_cuda,--as-needed)
endfunction()

lbtorch_fsdp_test(test_fsdp_training)
lbtorch_fsdp_test(test_baseline_single_gpu)
