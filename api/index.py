# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time
import psutil
from numba import njit
import math

app = Flask(__name__)
CORS(app)

# ---------------------------
# Numba-accelerated Haversine
# ---------------------------
@njit
def haversine_km(lat1, lon1, lat2, lon2):
    # Earth radius in km
    R = 6371.0
    # convert decimal degrees to radians
    phi1 = lat1 * (math.pi / 180.0)
    phi2 = lat2 * (math.pi / 180.0)
    dphi = (lat2 - lat1) * (math.pi / 180.0)
    dlambda = (lon2 - lon1) * (math.pi / 180.0)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))

@njit
def route_length_km_numba(coords):
    # coords: Nx2 float64 array where coords[:,0]=lat, coords[:,1]=lon
    n = coords.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n - 1):
        total += haversine_km(coords[i, 0], coords[i, 1], coords[i+1, 0], coords[i+1, 1])
    return total

# ---------------------------
# Helpers (non-numba)
# ---------------------------
def detect_and_fix_latlon_order(np_coords):
    """
    Heuristic: If many values in column 0 are outside [-90, 90] (invalid lat),
    assume coordinates are in order [lon, lat] and swap to [lat, lon].
    Returns fixed array and a flag indicating whether swap occurred.
    """
    col0 = np_coords[:, 0]
    invalid_lat_count = np.sum((col0 < -90) | (col0 > 90))
    # if more than half values are invalid as lat, swap columns
    if invalid_lat_count > (len(col0) // 2):
        fixed = np_coords[:, ::-1].copy()
        return fixed, True
    return np_coords, False

# ---------------------------
# Flask endpoint
# ---------------------------
@app.route('/ektruck/calculate', methods=['POST'])
def calculate_grouped_distances():
    start = time.time()
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    cpu_before = psutil.cpu_percent(interval=None)

    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON or empty body"}), 400

    # Accept both a single group object or list of groups
    groups = data if isinstance(data, list) else [data]

    results = []
    total_points = 0

    for group in groups:
        group_id = group.get("id", None)
        coords = group.get("coordinates", None)

        if coords is None:
            results.append({"id": group_id, "error": "no coordinates provided"})
            continue

        # Try to coerce into a Nx2 float64 numpy array
        try:
            np_coords = np.array(coords, dtype=np.float64)
        except Exception as e:
            results.append({"id": group_id, "error": f"invalid coordinate format: {str(e)}"})
            continue

        # Validate shape
        if np_coords.ndim != 2 or np_coords.shape[1] != 2:
            results.append({"id": group_id, "error": "coordinates must be list of [lat, lon] pairs"})
            continue

        # Fix ordering if necessary (lon,lat -> lat,lon)
        np_coords, swapped = detect_and_fix_latlon_order(np_coords)

        # compute route distance in km (consecutive points)
        try:
            dist_km = float(route_length_km_numba(np_coords))
        except Exception as e:
            # fallback to python implementation if numba compilation fails
            dist_km = 0.0
            for i in range(np_coords.shape[0] - 1):
                dist_km += _haversine_km_py(np_coords[i,0], np_coords[i,1], np_coords[i+1,0], np_coords[i+1,1])

        results.append({
            "id": group_id,
            "points_count": int(np_coords.shape[0]),
            "distance_km": round(dist_km, 6),
            "latlon_swapped_fixed": bool(swapped)
        })

        total_points += int(np_coords.shape[0])

    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)  # MB
    end = time.time()

    return jsonify({
        "cpu_percent_used": cpu_after,
        "peak_memory_mb": round(mem_after - mem_before, 4),
        "points_count": total_points,
        "results": results,
        "time_taken_sec": round(end - start, 6)
    })


# ---------------------------
# Pure-python fallback haversine (used only on exception)
# ---------------------------
def _haversine_km_py(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))

