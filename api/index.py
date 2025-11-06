# flask_haversine_parallel.py
from flask import Flask, request, jsonify
import numpy as np
import time
import psutil
import os
import math

app = Flask(__name__)

# Optional: try to import numba. If not available we'll fallback.
try:
    from numba import njit, prange, set_num_threads, get_num_threads
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# Configuration (tune these)
CHUNK_SIZE = int(os.getenv("HAVE_CHUNK_SIZE", 1_000_000))  # number of points per chunk
NUM_THREADS = int(os.getenv("NUMBA_NUM_THREADS", 0))  # 0 => let numba decide / use env var

# --- Haversine functions ---

if NUMBA_AVAILABLE:
    # Numba-accelerated single-leg haversine (radians conversion done inside)
    @njit(fastmath=True)
    def haversine_leg(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
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

    # Parallel sum of consecutive legs for a coordinate chunk
    @njit(parallel=True, fastmath=True)
    def haversine_consecutive_parallel_chunk(coords):
        # coords: np.ndarray shape (n,2) dtype=float64, lat, lon in degrees
        n = coords.shape[0]
        if n < 2:
            return 0.0
        legs = np.empty(n - 1, dtype=np.float64)
        for i in prange(n - 1):
            lat1 = coords[i, 0] * (math.pi / 180.0)
            lon1 = coords[i, 1] * (math.pi / 180.0)
            lat2 = coords[i + 1, 0] * (math.pi / 180.0)
            lon2 = coords[i + 1, 1] * (math.pi / 180.0)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (math.sin(dlat * 0.5) ** 2) + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon * 0.5) ** 2)
            c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
            legs[i] = 6371.0 * c
        total = 0.0
        for i in range(legs.shape[0]):
            total += legs[i]
        return total

else:
    # Fallback pure-Python haversine (slower)
    def haversine_leg(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
        R = 6371.0
        dlat = math.radians(lat2_deg - lat1_deg)
        dlon = math.radians(lon2_deg - lon1_deg)
        a = math.sin(dlat/2.0)**2 + math.cos(math.radians(lat1_deg)) * math.cos(math.radians(lat2_deg)) * math.sin(dlon/2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c

    def haversine_consecutive_parallel_chunk(coords):
        # coords: np.ndarray shape (n,2)
        n = coords.shape[0]
        if n < 2:
            return 0.0
        total = 0.0
        for i in range(n - 1):
            total += haversine_leg(coords[i,0], coords[i,1], coords[i+1,0], coords[i+1,1])
        return total

# --- Chunking helper (handles overlaps correctly) ---
def chunked_haversine_sum(np_coords, chunk_size=CHUNK_SIZE):
    """
    np_coords: np.ndarray shape (N,2) dtype=float64
    chunk_size: number of points per chunk (not legs). Eg 1_000_000
    returns: total distance in km (float)
    """
    n = np_coords.shape[0]
    if n < 2:
        return 0.0

    total = 0.0
    start = 0
    first_chunk = True

    while start < n:
        end = min(start + chunk_size, n)
        # If not first chunk, include previous point to compute the connecting leg,
        # then subtract that leg to avoid double-counting.
        if not first_chunk:
            chunk_start_idx = start - 1
        else:
            chunk_start_idx = start

        chunk = np_coords[chunk_start_idx:end]
        # chunk is a view; ensure contiguous float64
        chunk = np.ascontiguousarray(chunk, dtype=np.float64)

        # compute chunk sum (this includes the connecting leg if chunk_start_idx == start-1)
        chunk_sum = float(haversine_consecutive_parallel_chunk(chunk))

        if not first_chunk:
            # subtract the first leg of this chunk (which is the connecting leg already counted)
            # first leg in chunk is between chunk[0] and chunk[1]
            extra_leg = float(haversine_leg(chunk[0,0], chunk[0,1], chunk[1,0], chunk[1,1]))
            total += (chunk_sum - extra_leg)
        else:
            total += chunk_sum
            first_chunk = False

        start += chunk_size

    return total

# --- Warm-up / thread config ---
def warmup_numba():
    if not NUMBA_AVAILABLE:
        return
    # Set threads if requested
    if NUM_THREADS > 0:
        try:
            set_num_threads(NUM_THREADS)
        except Exception:
            pass
    # Call compiled functions once to trigger compilation
    sample = np.array([[12.9716, 77.5946], [12.9720, 77.5950]], dtype=np.float64)  # tiny sample
    try:
        # Call leg and chunk functions
        _ = haversine_leg(12.9716, 77.5946, 12.9720, 77.5950)
        _ = haversine_consecutive_parallel_chunk(sample)
    except Exception:
        # ignore compilation/runtime issues for warmup
        pass

# Run warmup at module import so server startup compiles Numba functions before first request
warmup_numba()

# --- Flask endpoint (same response format as you requested) ---
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

        # Ensure numpy array lat,lon float64
        np_coords = np.array(coords, dtype=np.float64)
        total_points += len(np_coords)

        # Compute distance using chunked parallel Numba function (or fallback)
        try:
            dist = float(chunked_haversine_sum(np_coords, chunk_size=CHUNK_SIZE))
        except Exception as e:
            # If something goes wrong with Numba or large memory, fallback to simple loop
            dist = 0.0
            for i in range(len(np_coords) - 1):
                dist += haversine_leg(np_coords[i, 0], np_coords[i, 1], np_coords[i + 1, 0], np_coords[i + 1, 1])

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
