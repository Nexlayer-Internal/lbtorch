#!/bin/bash
# CPU gradient gate of torch::models::nemotron_h vs HF autograd. No GPU. usage: run_gate.sh [build_dir] [fixture_dir]
set -e
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=${1:-/mnt/raid/unify/bg-lbtorch-nemotron-build}; FX=${2:-/mnt/raid/unify/bg-lbtorch-nemotron-fx}
REAL=/mnt/raid/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
/usr/bin/python3 $HERE/gen_fixture.py $FX        # HF transformers 5.13.1, fp64 + fp32 oracles (state-passing axis fixed)
cmake -S $HERE -B $BUILD -DCMAKE_BUILD_TYPE=Release && cmake --build $BUILD -j8
$BUILD/nemotron_h_check $FX $HERE/results_cpu.json $REAL
