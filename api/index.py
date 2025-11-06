from flask import Flask, request, jsonify
import numpy as np
import time
import psutil
from numba import njit, prange
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Haversine (kilometers)
@njit
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    # convert degrees to radians
    phi1 = lat1 * (np.pi / 180.0)
    phi2 = lat2 * (np.pi / 180.0)
    dphi = (lat2 - lat1) * (np.pi / 180.0)
    dlambda = (lon2 - lon1) * (np.pi / 180.0)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))

# Pairwise sum using haversine; parallelizable
@njit(parallel=True)
def pairwise_haversine_sum(coords):
    n = coords.shape[0]
    total = 0.0
    for i in prange(n):
        lat_i = coords[i, 0]
        lon_i = coords[i, 1]
        # j starts at i+1 to avoid double-count and self-distance
        for j in range(i + 1, n):
            total += haversine_km(lat_i, lon_i, coords[j, 0], coords[j, 1])
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

        # Ensure float dtype (important for numba)
        try:
            np_coords = np.array(coords, dtype=np.float64)
        except Exception as e:
            return jsonify({"error": f"Invalid coordinate format for group {group_id}: {e}"}), 400

        # validate shape
        if np_coords.ndim != 2 or np_coords.shape[1] != 2:
            return jsonify({"error": f"Coordinates for group {group_id} must be a list of [lat, lon] pairs."}), 400

        total_points += len(np_coords)
        dist_km = pairwise_haversine_sum(np_coords)  # result in kilometers
        results.append({
            "id": group_id,
            "distance_km": round(dist_km, 4)
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
