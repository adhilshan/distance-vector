from flask import Flask, request, jsonify
import numpy as np
from numba import njit, prange
import time
import psutil
import os
import tracemalloc

app = Flask(__name__)

# Numba-accelerated pairwise sum
@njit(parallel=True, fastmath=True)
def numba_pairwise_sum(coords):
    total = 0.0
    n = coords.shape[0]
    for i in prange(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            total += (dx * dx + dy * dy) ** 0.5
    return total

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        coords = np.array(data["coordinates"], dtype=np.float32)

        if coords.ndim != 2 or coords.shape[1] != 2:
            return jsonify({"error": "Coordinates must be a list of [x, y] pairs"}), 400

        process = psutil.Process(os.getpid())
        tracemalloc.start()

        # Warm-up
        numba_pairwise_sum(coords[:10])

        start_time = time.perf_counter()
        cpu_start = process.cpu_percent(interval=None)

        total = numba_pairwise_sum(coords)

        cpu_end = process.cpu_percent(interval=None)
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cpu_usage = (cpu_start + cpu_end) / 2
        ram_mb = peak / (1024 * 1024)

        return jsonify({
            "sum_of_distances": round(float(total), 2),
            "time_taken_sec": round(end_time - start_time, 4),
            "cpu_percent_used": round(cpu_usage, 1),
            "peak_memory_mb": round(ram_mb, 2),
            "points_count": len(coords)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "POST coordinates to /calculate as JSON with format: { 'coordinates': [[x1,y1], [x2,y2], ...] }"
