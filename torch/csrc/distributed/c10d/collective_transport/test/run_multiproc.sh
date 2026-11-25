#!/usr/bin/env bash
# Launches one process per rank for the collective_transport test binaries.
# Deliberately plain bash (no Python anywhere in this build or test path,
# per the project's zero-Python goal) -- each test binary itself reads
# RANK/WORLD_SIZE/MASTER_PORT plus whatever op-specific env vars it needs
# (see test_common.h and each test's own header comment) and does its own
# in-process rendezvous via TCPStore.
#
# Usage:
#   run_multiproc.sh <binary> <world_size> <port> <timeout_seconds>
#
# Any OTHER environment variables already exported in the calling shell
# (OP, SHARD_NUMEL, DTYPE, FORCE_TRANSPORT, NUM_LOCAL_GPUS, REPEAT, MODE,
# BARRIER_TIMEOUT_MS, ...) are passed through to every rank unchanged.
#
# Exit code: 0 only if every rank exited 0. `timeout_seconds` is a hard
# backstop (via coreutils `timeout --kill-after`) so a genuine hang in the
# binary under test (expected for repro_vanilla_nccl_hang.cpp) cannot hang
# this script indefinitely -- it is a backstop, not the primary safety
# mechanism (this transport layer's own bounded waits are), and its firing
# should be treated as a real finding, not a shrug.
set -u

BIN="$1"
WORLD_SIZE="$2"
PORT="$3"
TIMEOUT_SECONDS="${4:-120}"

PIDS=()
LOGDIR=$(mktemp -d)
echo "run_multiproc: launching ${WORLD_SIZE} ranks of ${BIN} on port ${PORT}, logs in ${LOGDIR}, timeout=${TIMEOUT_SECONDS}s"

for ((rank=0; rank<WORLD_SIZE; rank++)); do
  RANK="$rank" WORLD_SIZE="$WORLD_SIZE" MASTER_PORT="$PORT" \
    timeout --kill-after=10s "${TIMEOUT_SECONDS}s" "$BIN" \
    > "${LOGDIR}/rank${rank}.log" 2>&1 &
  PIDS+=("$!")
done

overall_rc=0
for ((rank=0; rank<WORLD_SIZE; rank++)); do
  # Do not negate `wait` directly in the `if` -- that collapses $? to the
  # if-condition's own 0/1 result instead of the real exit code. Capture it
  # first.
  wait "${PIDS[$rank]}"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "run_multiproc: rank ${rank} exited with code ${rc}"
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      echo "run_multiproc: rank ${rank} was KILLED BY THE TIMEOUT BACKSTOP (${TIMEOUT_SECONDS}s) -- this means the process under test hung"
    fi
    overall_rc=1
  fi
done

echo "---- per-rank output ----"
for ((rank=0; rank<WORLD_SIZE; rank++)); do
  echo "-- rank ${rank} (${LOGDIR}/rank${rank}.log) --"
  cat "${LOGDIR}/rank${rank}.log"
done
echo "--------------------------"

exit "$overall_rc"
