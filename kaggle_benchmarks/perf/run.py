import json
import statistics
import subprocess
import sys
import time

# Pin torch 2.5.1+cu121 FIRST: Kaggle's default torch 2.10 only supports sm_70+ and
# crashes on Tesla P100 (sm_60) with cudaErrorNoKernelImageForDevice.
# torch 2.5.1+cu121 supports sm_60–sm_90, so it works on P100 AND T4.
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "torch==2.5.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    ]
)

# Install cayleypy's non-torch runtime deps separately so the next step can use
# --no-deps: cayleypy declares torch>=2.6.0, but it runs correctly on 2.5.1, and
# --no-deps keeps pip from re-resolving torch upward above the 2.5.1 pin (which would
# both waste a large download and emit a dependency-conflict warning). Kaggle
# preinstalls numpy/scipy; h5py/numba/kagglehub are ensured here.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "h5py", "numba", "kagglehub"])

# Install cayleypy from the perf branch. Pinned to commit SHA `be8b087` for
# reproducible final validation (Phase 1 + Phase 4 + bugfix + Task 3.2).
# To resume auto-tracking the branch, replace the SHA with `feature/beam-search-perf`.
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "git+https://github.com/cayleypy/cayleypy.git@be8b087",
    ]
)

import torch
from cayleypy import CayleyGraph, PermutationGroups, Puzzles

# Fix ALL randomness for reproducible before/after comparison across kernel versions.
# The hasher uses random.randint for its seed when none is passed; without fixing it,
# each kernel run gets a different hash function, so path_found/path_length/depth-run
# timing vary stochastically (a v1-vs-v2 deep-run difference of "found at step 43" vs
# "not found in 100" is hash-seed noise, NOT a regression). This matches conftest.py's
# DETERMINISTIC_SEED=12345 used by the CPU test suite.
import random

import numpy as np

np.random.seed(12345)
random.seed(12345)
torch.manual_seed(12345)

# --- Scenario constants (keep in sync with cayleypy/algo/beam_search_benchmark.py) ---
_LRX8_START = [3, 5, 7, 1, 0, 6, 4, 2]
_BEAM_WIDTH_QUICK = 10**5
_MAX_STEPS_QUICK = 30  # MUST match beam_search_benchmark.py

# Cube 3x3x3 QTM scramble from ivankolt/hamming-beamsearch.
# fmt: off
_CUBE333_START = [3, 3, 1, 0, 0, 2, 1, 0, 4, 4, 2, 0, 5, 1, 4, 5, 5, 3,
                  3, 3, 5, 0, 2, 5, 4, 2, 0, 2, 4, 0, 2, 3, 3, 2, 5, 5,
                  2, 1, 0, 0, 4, 1, 2, 4, 1, 4, 3, 5, 1, 5, 1, 3, 4, 1]
# fmt: on
_BEAM_WIDTH_DEEP = 2**18
_MAX_STEPS_DEEP = 100

# --- Large-group scenario constants (Phase 0.3 profiling; keep in sync with
#     cayleypy/algo/beam_search_benchmark.py). Start states are deterministic
#     scrambles generated once via random_walks(seed=12345) and hardcoded here so
#     the Kaggle kernel does not depend on import-time RNG.
#     Scramble lengths are short (10-12 moves) so the search finds the solution
#     within max_steps=30 using only the hamming predictor (no NN model). ---

# LRX(32): last 16 elements rotated to the front (16 L-moves from central).
_LRX32_START = list(range(16, 32)) + list(range(0, 16))

# cube444: 12-move scramble (solution length 9 with hd=2).
# fmt: off
_CUBE444_START = [1, 5, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 4, 0, 0, 0, 5, 1,
                  4, 4, 1, 4, 1, 1, 2, 1, 1, 2, 4, 1, 4, 4, 1, 5, 5, 2,
                  2, 5, 5, 2, 3, 3, 2, 0, 1, 1, 1, 1, 3, 3, 4, 2, 3, 3,
                  2, 2, 4, 3, 4, 4, 2, 5, 3, 2, 0, 0, 3, 3, 0, 0, 3, 4,
                  1, 0, 1, 1, 3, 3, 3, 0, 3, 0, 3, 5, 5, 5, 5, 5, 5, 4,
                  4, 5, 5, 4, 4, 5]
# fmt: on
# cube555: 10-move scramble (solution length 7 with hd=2).
# fmt: off
_CUBE555_START = [5, 5, 5, 5, 5, 2, 2, 2, 0, 2, 3, 0, 0, 0, 0, 2, 2, 2,
                  2, 2, 3, 0, 0, 0, 0, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
                  1, 1, 1, 1, 3, 3, 3, 3, 3, 0, 1, 1, 1, 1, 2, 5, 2, 5,
                  4, 2, 5, 2, 5, 4, 2, 5, 2, 5, 4, 2, 4, 4, 5, 4, 2, 1,
                  2, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  0, 1, 1, 1, 1, 5, 5, 5, 5, 4, 2, 0, 0, 0, 0, 2, 0, 4,
                  0, 4, 2, 0, 4, 0, 4, 2, 0, 2, 2, 4, 3, 3, 4, 3, 4, 1,
                  5, 5, 5, 5, 0, 4, 4, 4, 4, 1, 5, 5, 5, 5, 0, 4, 4, 5,
                  4, 0, 4, 4, 4, 4]
# fmt: on

# Profiling scenarios (Phase 0.3): bw=2^18, iterated mode, hd=2, 30 steps, 1 run each.
# Captures verbose=100 per-region timing breakdown for the 4 groups listed in the plan.
_BEAM_WIDTH_PROFILE = 2**18
_MAX_STEPS_PROFILE = 30

graph_lrx8 = CayleyGraph(PermutationGroups.lrx(8))
graph_lrx32 = CayleyGraph(PermutationGroups.lrx(32))
graph_cube333 = CayleyGraph(
    Puzzles.rubik_cube(3, metric="QTM"),
    dtype=torch.int8,
    bit_encoding_width=None,
    hash_chunk_size=2**16,
)
graph_cube444 = CayleyGraph(
    Puzzles.rubik_cube(4, metric="QTM"),
    dtype=torch.int8,
    bit_encoding_width=None,
    hash_chunk_size=2**16,
)
graph_cube555 = CayleyGraph(
    Puzzles.rubik_cube(5, metric="QTM"),
    dtype=torch.int8,
    bit_encoding_width=None,
    hash_chunk_size=2**16,
)
device = graph_lrx8.device
print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)

QUICK_MODES = [
    {"name": "simple"},
    {"name": "advanced", "history_depth": 2},
    {"name": "iterated", "history_depth": 2},
    {"name": "iterated_batched", "history_depth": 2},
]


def run_benchmark(graph, start_state, modes, beam_width, max_steps, warmup=2, measured=5):
    """Run benchmark for each mode, return dict of stats."""
    out = {}
    for mode in modes:
        kwargs = {"beam_mode": mode["name"], "beam_width": beam_width, "max_steps": max_steps}
        if "history_depth" in mode:
            kwargs["history_depth"] = mode["history_depth"]
        for _ in range(warmup):
            graph.beam_search(start_state=start_state, **kwargs)
        times = []
        last = None
        for _ in range(measured):
            if device.type == "cuda":
                torch.cuda.synchronize()  # CRITICAL: flush queued GPU kernels before timing
            t0 = time.time()
            last = graph.beam_search(start_state=start_state, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize()  # CRITICAL: wait for all GPU work to finish
            times.append(time.time() - t0)
        out[mode["name"]] = {
            "min": min(times),
            "mean": statistics.mean(times),
            "stddev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "path_found": last.path_found,
            "path_length": last.path_length,
        }
        print(
            f"  {mode['name']}: min={min(times):.3f}s mean={statistics.mean(times):.3f}s "
            f"found={last.path_found} len={last.path_length}",
            flush=True,
        )
    return out


def run_profiling(graph, name, start_state, beam_width, max_steps, history_depth=2, beam_mode="iterated"):
    """Run ONE beam_search pass with verbose=100 and capture the per-step
    profiling lines from stdout. verbose=100 adds torch.cuda.synchronize() brackets
    around timed regions (see beam_search.py Task 0.2), so the printed t_hash/t_sort/
    t_dedup/t_check/t_predict/t_isin/t_moves values reflect actual GPU execution time,
    not kernel launch time. Runs once (not measured) — the sync cost would pollute
    timings if run in the benchmark loop.
    """
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    kwargs = {
        "start_state": start_state,
        "beam_mode": beam_mode,
        "beam_width": beam_width,
        "max_steps": max_steps,
        "verbose": 100,
        "return_path": False,
    }
    # simple mode has no history_depth param.
    if beam_mode != "simple":
        kwargs["history_depth"] = history_depth
    with redirect_stdout(buf):
        result = graph.beam_search(**kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    profile_lines = buf.getvalue().splitlines()
    print(
        f"  {name} [{beam_mode}]: {elapsed:.2f}s, found={result.path_found}, len={result.path_length}, "
        f"{len(profile_lines)} profile lines",
        flush=True,
    )
    # Echo the last profiling line so per-region attribution is visible in the Kaggle log.
    for line in profile_lines[-3:]:
        print(f"    {line}", flush=True)
    return {
        "elapsed_sec": elapsed,
        "path_found": result.path_found,
        "path_length": result.path_length,
        "profile_lines": profile_lines,
    }


results = {}

# --- Quick level: LRX(8) ---
print("=== LRX(8) quick ===", flush=True)
results["lrx8"] = run_benchmark(graph_lrx8, _LRX8_START, QUICK_MODES, _BEAM_WIDTH_QUICK, _MAX_STEPS_QUICK)

# --- Quick level: cube 3x3x3 ---
print("=== Cube 3x3x3 quick ===", flush=True)
results["cube333_quick"] = run_benchmark(
    graph_cube333, _CUBE333_START, QUICK_MODES, _BEAM_WIDTH_QUICK, _MAX_STEPS_QUICK
)

# --- Deep level: cube 3x3x3 (iterated only, notebook params, 1 run) ---
# Expensive (~20 min on P100). Comment out the block below to skip.
print("=== Cube 3x3x3 deep ===", flush=True)
deep_kwargs = {
    "beam_mode": "iterated",
    "beam_width": _BEAM_WIDTH_DEEP,
    "max_steps": _MAX_STEPS_DEEP,
    "history_depth": 2,
    "hashed_neigbourhood": 3,
    "return_path": False,
}
if device.type == "cuda":
    torch.cuda.synchronize()
deep_t0 = time.time()
deep_result = graph_cube333.beam_search(start_state=_CUBE333_START, **deep_kwargs)
if device.type == "cuda":
    torch.cuda.synchronize()
deep_time = time.time() - deep_t0
results["cube333_deep"] = {
    "time_sec": deep_time,
    "path_found": deep_result.path_found,
    "path_length": deep_result.path_length,
}
print(f"  iterated: time={deep_time:.1f}s found={deep_result.path_found}", flush=True)

# --- Deep level: cube 3x3x3 iterated_batched (regime B, 1 run) ---
# Compares batched vs chunked iterated at deep beam (bw=2^18, mitm=3).
print("=== Cube 3x3x3 deep (iterated_batched) ===", flush=True)
deep_batched_kwargs = {
    "beam_mode": "iterated_batched",
    "beam_width": _BEAM_WIDTH_DEEP,
    "max_steps": _MAX_STEPS_DEEP,
    "history_depth": 2,
    "hashed_neigbourhood": 3,
    "return_path": False,
}
if device.type == "cuda":
    torch.cuda.synchronize()
deep_b_t0 = time.time()
deep_b_result = graph_cube333.beam_search(start_state=_CUBE333_START, **deep_batched_kwargs)
if device.type == "cuda":
    torch.cuda.synchronize()
deep_b_time = time.time() - deep_b_t0
results["cube333_deep_batched"] = {
    "time_sec": deep_b_time,
    "path_found": deep_b_result.path_found,
    "path_length": deep_b_result.path_length,
}
print(f"  iterated_batched: time={deep_b_time:.1f}s found={deep_b_result.path_found}", flush=True)

# --- Phase 0.3 profiling: large groups, iterated mode, verbose=100 ---
# One-shot per group (not measured). Captures per-region GPU timing breakdown to
# attribute time to t_hash/t_sort/t_dedup/t_check/t_predict/t_isin/t_moves. The
# sync brackets added in Task 0.2 make these numbers reflect actual GPU execution.
print("=== Phase 0.3 profiling (verbose=100) ===", flush=True)
results["profiling"] = {}
results["profiling"]["cube333"] = run_profiling(
    graph_cube333, "cube333", _CUBE333_START, _BEAM_WIDTH_PROFILE, _MAX_STEPS_PROFILE
)
results["profiling"]["cube444"] = run_profiling(
    graph_cube444, "cube444", _CUBE444_START, _BEAM_WIDTH_PROFILE, _MAX_STEPS_PROFILE
)
results["profiling"]["cube555"] = run_profiling(
    graph_cube555, "cube555", _CUBE555_START, _BEAM_WIDTH_PROFILE, _MAX_STEPS_PROFILE
)
results["profiling"]["lrx32"] = run_profiling(
    graph_lrx32, "lrx32", _LRX32_START, _BEAM_WIDTH_PROFILE, _MAX_STEPS_PROFILE
)

# --- Advanced + simple mode profiling (Task 1: profile non-iterated modes) ---
# Advanced uses get_unique_states (bundles hash+sort+dedup) and nonbacktrack once
# per step (vs n_gens× per step in iterated). Simple has no nonbacktrack at all.
# These runs reveal whether the same bottlenecks apply, or if different regions dominate.
print("=== Advanced mode profiling (verbose=100) ===", flush=True)
results["profiling_advanced"] = {}
results["profiling_advanced"]["cube333"] = run_profiling(
    graph_cube333, "cube333", _CUBE333_START, _BEAM_WIDTH_PROFILE, _MAX_STEPS_PROFILE, beam_mode="advanced"
)
results["profiling_advanced"]["cube555"] = run_profiling(
    graph_cube555, "cube555", _CUBE555_START, _BEAM_WIDTH_PROFILE, _MAX_STEPS_PROFILE, beam_mode="advanced"
)

print("=== Simple mode profiling (verbose=100) ===", flush=True)
results["profiling_simple"] = {}
results["profiling_simple"]["cube333"] = run_profiling(
    graph_cube333, "cube333", _CUBE333_START, _BEAM_WIDTH_PROFILE, _MAX_STEPS_PROFILE, beam_mode="simple"
)

# --- Phase 0.3 memory probe: cube555 at 2^22 ---
# Confirms peak GPU memory at the largest target beam that should fit BEFORE the
# Phase 1 compaction tasks. After Task 1.6 (compact nonbacktrack), this same probe
# validates the 6 GB -> 0.25 GB reduction. Skippable by commenting the block below.
if device.type == "cuda":
    print("=== cube555 memory probe (bw=2^22) ===", flush=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    probe_t0 = time.time()
    probe_result = graph_cube555.beam_search(
        start_state=_CUBE555_START,
        beam_mode="iterated",
        beam_width=2**22,
        max_steps=10,
        history_depth=2,
        return_path=False,
    )
    torch.cuda.synchronize()
    probe_time = time.time() - probe_t0
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    results["cube555_mem_probe_2_22"] = {
        "time_sec": probe_time,
        "peak_memory_gb": peak_gb,
        "path_found": probe_result.path_found,
        "path_length": probe_result.path_length,
    }
    print(
        f"  cube555 bw=2^22: {probe_time:.1f}s, peak={peak_gb:.2f} GB, " f"found={probe_result.path_found}",
        flush=True,
    )

results["meta"] = {
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "branch": "feature/beam-search-perf",
}
with open("gpu_benchmark_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
