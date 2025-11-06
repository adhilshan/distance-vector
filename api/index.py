from flask import Flask, request, jsonify
import numpy as np
import time
import psutil
from numba import njit
from flask_cors import CORS
import math

app = Flask(__name__)
CORS(app)

# NOTE: keeping original function name so you don't have to change calls elsewhere.
# This now computes the route distance (sum of consecutive haversine distances in KM)
# for GPS coords (lat, lon) provided in the input.
@njit
def pairwise_distance_sum(coords):
    """
    coords: Nx2 float64 numpy array with [lat, lon] in decimal degrees.
    Returns route length in kilometers (sum of distances between consecutive points).
    """
    n = coords.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    R = 6371.0  # Earth radius in km
    for i in range(n - 1):
        lat1 = coords[i, 0] * (math.pi / 180.0)
        lon1 = coords[i, 1] * (math.pi / 180.0)
        lat2 = coords[i + 1, 0] * (math.pi / 180.0)
        lon2 = coords[i + 1, 1] * (math.pi / 180.0)

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
        # safe clamp for numeric stability
        c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
        total += R * c
    return total

def detect_and_fix_latlon_order(np_coords):
    """
    Heuristic to detect if coords are [lon, lat] instead of [lat, lon].
    If more than half values in column 0 are outside [-90, 90], swap columns.
    Returns (fixed_array, swapped_flag)
    """
    col0 = np_coords[:, 0]
    invalid_lat_count = np.sum((col0 < -90.0) | (col0 > 90.0))
    if invalid_lat_count > (len(col0) // 2):
        # swap columns
        fixed = np_coords[:, ::-1].copy()
        return fixed, True
    return np_coords, False

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

        # Force float64 (important for Numba) and validate shape
        try:
            np_coords = np.array(coords, dtype=np.float64)
        except Exception as e:
            results.append({
                "id": group_id,
                "distance": None,
                "error": f"invalid coordinate format: {e}"
            })
            continue

        if np_coords.ndim != 2 or np_coords.shape[1] != 2:
            results.append({
                "id": group_id,
                "distance": None,
                "error": "coordinates must be Nx2 array of [lat, lon]"
            })
            continue

        # Fix ordering if client sent [lon, lat]
        np_coords, swapped = detect_and_fix_latlon_order(np_coords)

        # compute route distance (consecutive points) in kilometers using numba
        try:
            dist_km = pairwise_distance_sum(np_coords)
        except Exception as e:
            # In case Numba compilation fails at runtime, fallback to a safe python haversine
            dist_km = 0.0
            for i in range(np_coords.shape[0] - 1):
                lat1, lon1 = np_coords[i]
                lat2, lon2 = np_coords[i+1]
                # pure-python haversine (km)
                phi1 = math.radians(lat1)
                phi2 = math.radians(lat2)
                dphi = math.radians(lat2 - lat1)
                dlambda = math.radians(lon2 - lon1)
                a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
                c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
                dist_km += 6371.0 * c

        total_points += int(np_coords.shape[0])
        results.append({
            "id": group_id,
            # keep same key name as original; distance now in kilometers
            "distance": round(float(dist_km), 4)
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
