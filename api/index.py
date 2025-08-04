from flask import Flask, request, jsonify
import numpy as np
import time
import psutil
from numba import njit

app = Flask(__name__)
# Optimized distance sum using Numba
@njit
def pairwise_distance_sum(coords):
    n = len(coords)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            total += (dx * dx + dy * dy) ** 0.5
    return total

@app.route('/ektruck/calculate', methods=['POST'])
def calculate_grouped_distances():
    data = request.get_json()

    if not isinstance(data, list):
        return jsonify({"error": "Input should be a list of groups with 'id' and 'coordinates'"}), 400

    process = psutil.Process()
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB

    start = time.time()
    results = []
    total_points = 0

    for group in data:
        group_id = group.get("id")
        coords = group.get("coordinates")

        if not group_id or not coords:
            continue

        np_coords = np.array(coords)
        total_points += len(np_coords)
        dist = pairwise_distance_sum(np_coords)
        results.append({
            "id": group_id,
            "distance": round(dist, 4)
        })

    end = time.time()
    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)  # MB

    return jsonify({
        "cpu_percent_used": cpu_after,
        "peak_memory_mb": round(mem_after - mem_before, 4),
        "points_count": total_points,
        "sum_of_distances": results,
        "time_taken_sec": round(end - start, 4)
    })
