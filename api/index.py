# flask_haversine_njit_singlethread.py
from flask import Flask, request, jsonify
import numpy as np
import time
import psutil
import os
import math
import warnings

app = Flask(__name__)

# Try to import numba; if not available fallback to Python-only
try:
    from numba import njit, set_num_threads
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# Configuration
CHUNK_SIZE = int(os.getenv("HAVE_CHUNK_SIZE", 1_000_000))  # points per chunk
NUMBA_NUM_THREADS_ENV = os.getenv("NUMBA_NUM_THREADS", None)

# If running in restricted environment (like Vercel), force single-threaded Numba if possible.
if NUMBA_AVAILABLE:
    try:
        # If the user or environment set a thread count, use it, else force 1 to avoid parallel runtime.
        if NUMBA_NUM_THREADS_ENV is not None:
            set_num_threads(int(NUMBA_NUM_THREADS_ENV))
        else:
            # Force single-threaded to avoid semaphore/shared mem issues on platforms like Vercel
            set_num_threads(1)
    except Exception:
        # ignore if we can't set threads
        pass

# --- Haversine implementations ---

if NUMBA_AVAILABLE:
    # Single-threaded numba-compiled haversine for a pair of points
    @njit(fastmath=True)
    def haversine_leg_njit(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
        R = 6371.0
        lat1 = lat1_deg * (math.pi / 180.0)
        lon1 = lon1_deg * (math.pi / 180.0)
        lat2 = lat2_deg * (math.pi / 180.0)
        lon2 = lon2_deg * (math.pi / 180.0)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat * 0.5) ** 2) + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon * 0.5) ** 2)
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c

    # Single-threaded numba-compiled consecutive-leg sum (no prange)
    @njit(fastmath=True)
    def haversine_consecutive_njit(coords):
        # coords: 2D array shape (n,2) dtype=float64 (lat, lon)
        n = coords.shape[0]
        if n < 2:
            return 0.0
        total = 0.0
        for i in range(n - 1):
            lat1 = coords[i, 0]
            lon1 = coords[i, 1]
            lat2 = coords[i + 1, 0]
            lon2 = coords[i + 1, 1]
            # convert and compute
            lat1_r = lat1 * (math.pi / 180.0)
            lon1_r = lon1 * (math.pi / 180.0)
            lat2_r = lat2 * (math.pi / 180.0)
            lon2_r = lon2 * (math.pi / 180.0)
            dlat = lat2_r - lat1_r
            dlon = lon2_r - lon1_r
            a = (math.sin(dlat * 0.5) ** 2) + math.cos(lat1_r) * math.cos(lat2_r) * (math.sin(dlon * 0.5) ** 2)
            c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
            total += 6371.0 * c
        return total
else:
    # Pure-Python fallback
    def haversine_leg_njit(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
        R = 6371.0
        dlat = math.radians(lat2_deg - lat1_deg)
        dlon = math.radians(lon2_deg - lon1_deg)
        a = math.sin(dlat/2.0)**2 + math.cos(math.radians(lat1_deg)) * math.cos(math.radians(lat2_deg)) * math.sin(dlon/2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c

    def haversine_consecutive_njit(coords):
        n = coords.shape[0]
        if n < 2:
            return 0.0
        total = 0.0
        for i in range(n - 1):
            total += haversine_leg_njit(coords[i,0], coords[i,1], coords[i+1,0], coords[i+1,1])
        return total

# --- Chunking helper (overlap handling) ---
def chunked_haversine_sum(np_coords, chunk_size=CHUNK_SIZE):
    """
    Sum consecutive haversine distances for very large coordinate arrays by chunking.
    Handles overlaps so legs at chunk boundaries are counted exactly once.
    np_coords: ndarray (N,2) dtype=float64
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
        # ensure contiguous float64 array for Numba
        chunk = np.ascontiguousarray(chunk, dtype=np.float64)

        # compute chunk sum via Numba or fallback
        chunk_sum = float(haversine_consecutive_njit(chunk))

        if not first_chunk:
            # subtract the connecting leg (first leg in this chunk) to avoid double-count
            extra_leg = float(haversine_leg_njit(chunk[0,0], chunk[0,1], chunk[1,0], chunk[1,1]))
            total += (chunk_sum - extra_leg)
        else:
            total += chunk_sum
            first_chunk = False

        start += chunk_size

    return total

# --- Warmup compile to avoid first-request latency ---
def warmup():
    # If Numba available, call compiled functions once with tiny data to trigger JIT compile.
    if NUMBA_AVAILABLE:
        try:
            sample = np.array([[12.9715987, 77.594566], [12.9722000, 77.595000]], dtype=np.float64)
            # Call functions to compile
            _ = haversine_leg_njit(12.9715987, 77.594566, 12.9722000, 77.595000)
            _ = haversine_consecutive_njit(sample)
        except Exception:
            # ignore warmup errors - safe fallback will run
            warnings.warn("Numba warmup failed (ignored). Running fallback if needed.", RuntimeWarning)

warmup()

# --- Flask endpoint: same JSON response shape you asked for ---
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

        np_coords = np.array(coords, dtype=np.float64)
        total_points += len(np_coords)

        try:
            dist = float(chunked_haversine_sum(np_coords, chunk_size=CHUNK_SIZE))
        except Exception:
            # Safe fallback: simple loop (pure Python)
            dist = 0.0
            for i in range(len(np_coords) - 1):
                dist += haversine_leg_njit(np_coords[i,0], np_coords[i,1], np_coords[i+1,0], np_coords[i+1,1])

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

if __name__ == '__main__':
    # Nothing special required for Vercel-like envs; just run
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)), threaded=True)
