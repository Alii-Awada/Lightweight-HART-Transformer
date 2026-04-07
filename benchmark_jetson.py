#!/usr/bin/env python
# coding: utf-8

import time

import numpy as np
import tensorflow as tf


def benchmark_tflite(model_path, input_shape, n_runs=1000):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    if inp["dtype"] == np.int8:
        dummy = np.random.randint(-128, 127, size=(1, *input_shape), dtype=np.int8)
    else:
        dummy = np.random.randn(1, *input_shape).astype(inp["dtype"])

    # Warmup
    for _ in range(50):
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
        interp.get_tensor(out["index"])

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
        interp.get_tensor(out["index"])
        latencies.append((time.perf_counter() - t0) * 1000.0)

    lat = np.array(latencies, dtype=np.float32)
    budget_ms = 640.0  # 64-sample step @ 100 Hz = 640 ms realtime budget
    print(f"Mean:  {lat.mean():.2f} ms")
    print(f"P95:   {np.percentile(lat, 95):.2f} ms")
    print(f"Real-time? {'YES' if np.percentile(lat, 95) < budget_ms else 'NO'}")
    print(f"Throughput: {1000.0 / lat.mean():.1f} inferences/sec")


if __name__ == "__main__":
    raise SystemExit(
        "Import benchmark_tflite(...) from this module or call it from a short driver script."
    )

# Usage:
# benchmark_tflite("HART_int8.tflite", input_shape=(128, 6))
# Pair with: sudo tegrastats --interval 100 --logfile power.log
