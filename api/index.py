# flask_haversine_numpy.py
from flask import Flask, request, jsonify
import numpy as np
import time
import psutil
import os
import math
import warnings

app = Flask(__name__)

# Tunables
CHUNK_SIZE = int(os.getenv("HAVE_CHUNK_SIZE", 500_000))  # points per chunk; reduce if memory is tight
USE_FLOAT32 = False  # set True to use float32 (saves memory, slightly less precision)

# --- Vectorized haversine chunk implementation (NumPy only) ---
def haversine_consecutive_vectorized_chunk(coords):
    """
    coords: np.ndarray shape (n,2) dtype float64 (lat, lon in degrees)
    returns: total distance in kilometers (float)
    This function computes distances between consecutive points using vectorized numpy ops.
    """
    n = coords.shape[0]
    if n < 2:
        return 0.0

    # Convert to radians
    rad = np.radians(coords)  # shape (n,2)
    lat = rad[:, 0]
    lon = rad[:, 1]

    # Compute deltas for consecutive pairs
    lat1 = lat[:-1]
    lon1 = lon[:-1]
    lat2 = lat[1:]
    lon2 = lon[1:]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    sin_dlat2 = np.sin(dlat * 0.5)
    sin_dlon2 = np.sin(dlon * 0.5)

    a = sin_dlat2 * sin_dlat2 + np.cos(lat1) * np.cos(lat2) * (sin_dlon2 * sin_dlon2)
    # clip 'a' to [0,1] to avoid small numerical issues
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    R = 6371.0
    legs = R * c  # per-leg distances (array length n-1)

    return float(np.sum(legs))


def chunked_haversine_sum_np(np_coords, chunk_size=CHUNK_SIZE):
    """
    Sum consecutive haversine distances for very large coordinate arrays by chunking.
    Handles overlaps so legs at chunk boundaries are counted exactly once.

    np_coords: ndarray (N,2) dtype float64 (lat,lon)
    """
    n = np_coords.shape[0]
    if n < 2:
        return 0.0

    total = 0.0
    start = 0
    first_chunk = True

    while start < n:
        end = min(start + chunk_size, n)
        # include previous point as overlap for non-first chunks
        if not first_chunk:
            chunk_start_idx = start - 1
        else:
            chunk_start_idx = start

        chunk = np_coords[chunk_start_idx:end]
        # ensure contiguous and desired dtype
        dtype = np.float32 if USE_FLOAT32 else np.float64
        chunk = np.ascontiguousarray(chunk, dtype=dtype)

        chunk_sum = float(haversine_consecutive_vectorized_chunk(chunk))

        if not first_chunk:
            # subtract the connecting leg (first leg in this chunk) to avoid double-counting
            # compute connecting leg between chunk[0] and chunk[1]
            lat1, lon1 = float(chunk[0,0]), float(chunk[0,1])
            lat2, lon2 = float(chunk[1,0]), float(chunk[1,1])
            extra_leg = haversine_scalar(lat1, lon1, lat2, lon2)
            total += (chunk_sum - extra_leg)
        else:
            total += chunk_sum
            first_chunk = False

        start += chunk_size

    return total

# Small scalar haversine helper used only for the overlap correction
def haversine_scalar(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    R = 6371.0
    dlat = math.radians(lat2_deg - lat1_deg)
    dlon = math.radians(lon2_deg - lon1_deg)
    a = math.sin(dlat/2.0)**2 + math.cos(math.radians(lat1_deg)) * math.cos(math.radians(lat2_deg)) * math.sin(dlon/2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

# --- Warmup (not necessary but keeps first request predictable) ---
def warmup_numpy():
    # do a tiny vectorized call so imports and caches are ready
    sample = np.array([[12.9715987, 77.594566], [12.9722000, 77.595000]], dtype=np.float64)
    try:
        _ = haversine_consecutive_vectorized_chunk(sample)
    except Exception:
        warnings.warn("NumPy warmup failed (ignored).", RuntimeWarning)

warmup_numpy()

# --- Flask endpoint: same JSON response format ---
@app.route('/ektruck/calculate', methods=['POST'])
def calculate_grouped_distances():
    data = request.get_json()

    if not isinstance(data, list):
        return jsonify({"error": "Input should be a list of groups with 'id' and 'coordinates'"}), 400

    process = psutil.Process()
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB

    start_time = time.time()
    results = []
    total_points = 0

    for group in data:
        group_id = group.get("id")
        coords = group.get("coordinates")

        if not group_id or not coords:
            continue

        # Ensure numpy array lat,lon float64 (or float32 if configured)
        dtype = np.float32 if USE_FLOAT32 else np.float64
        np_coords = np.array(coords, dtype=dtype)
        total_points += len(np_coords)

        try:
            dist = float(chunked_haversine_sum_np(np_coords, chunk_size=CHUNK_SIZE))
        except Exception as e:
            # Last-resort fallback: simple Python loop
            dist = 0.0
            for i in range(len(np_coords) - 1):
                dist += haversine_scalar(float(np_coords[i,0]), float(np_coords[i,1]),
                                         float(np_coords[i+1,0]), float(np_coords[i+1,1]))

        results.append({
            "id": group_id,
            "distance": round(dist, 4)
        })

    end_time = time.time()
    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)  # MB

    return jsonify({
        "cpu_percent_used": cpu_after,
        "peak_memory_mb": round(mem_after - mem_before, 4),
        "points_count": total_points,
        "sum_of_distances": results,
        "time_taken_sec": round(end_time - start_time, 4)
    })
