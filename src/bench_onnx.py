"""Benchmark one ONNX model in a fresh process (called by the pipeline).

Usage: python -m src.bench_onnx model.onnx
Prints a JSON line with mean / p95 / p99 latency (µs) and memory increase (MB).
Protocol: one intra-op thread, batch size 1, pinned to one CPU core,
300 warm-up calls, 5,000 timed calls, repeated three times (median of each statistic).
"""
import json
import os
import sys
import time

import numpy as np
import psutil


def main(path, n_warm=300, n_timed=5000, repeats=3):
    try:
        os.sched_setaffinity(0, {0})
    except (AttributeError, OSError):
        pass  # not available on Windows / macOS
    import onnxruntime as ort

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    rss1 = proc.memory_info().rss
    name = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(n_timed, 5)).astype(np.float32)
    xs[:, 4] = (xs[:, 4] > 0.7).astype(np.float32)

    stats = []
    for _ in range(repeats):
        for i in range(n_warm):
            sess.run(None, {name: xs[i:i + 1]})
        t = np.empty(n_timed)
        for i in range(n_timed):
            t0 = time.perf_counter_ns()
            sess.run(None, {name: xs[i:i + 1]})
            t[i] = (time.perf_counter_ns() - t0) / 1000.0
        stats.append((t.mean(), np.percentile(t, 95), np.percentile(t, 99)))
    med = np.median(np.array(stats), axis=0)
    print(json.dumps(dict(mean_us=float(med[0]), p95_us=float(med[1]), p99_us=float(med[2]),
                          ram_mb=(rss1 - rss0) / 2**20,
                          size_kb=os.path.getsize(path) / 1024)))


if __name__ == "__main__":
    main(sys.argv[1])
