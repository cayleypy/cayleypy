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

# Install cayleypy pinned to an immutable commit SHA (NOT the mutable branch ref) so
# the baseline is reproducible even if the branch is force-pushed or deleted.
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "git+https://github.com/cayleypy/cayleypy.git@4ba6b0448861d6d5264bc021b2d67205b1fafbe3",
    ]
)

import torch
from cayleypy import CayleyGraph, PermutationGroups, Puzzles

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

graph_lrx8 = CayleyGraph(PermutationGroups.lrx(8))
graph_cube333 = CayleyGraph(
    Puzzles.rubik_cube(3, metric="QTM"),
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

results["meta"] = {
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "branch": "feature/foundation-perf-readiness",
}
with open("gpu_benchmark_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
