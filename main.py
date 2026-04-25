import hashlib
import json
import threading
from datetime import datetime, timedelta
from typing import Any, Dict

from fastapi import FastAPI, BackgroundTasks, HTTPException

from schemas.predict_schema import PredictRequest, PredictResponse
from services.model_service import run_iterative_forecast
from services.nasa_service import fetch_nasa_data

app = FastAPI(
    title="Solar Prediction API",
    version="2.0.0",
    description="Async Solar Prediction API with job tracking",
)

# ─── In-memory job store ────────────────────────────────────────────────────
# Structure: { job_id: { status, result, error, created_at } }
_jobs: Dict[str, Any] = {}
_jobs_lock = threading.Lock()

# ─── NASA response cache ─────────────────────────────────────────────────────
# Caches NASA data for 24 h to avoid redundant 25-year fetches
_nasa_cache: Dict[str, Any] = {}
_nasa_cache_lock = threading.Lock()
NASA_CACHE_TTL = timedelta(hours=24)


def _nasa_cache_key(lat: float, lon: float) -> str:
    return f"{round(lat, 4)}:{round(lon, 4)}"


def _get_cached_nasa(lat: float, lon: float):
    key = _nasa_cache_key(lat, lon)
    with _nasa_cache_lock:
        entry = _nasa_cache.get(key)
        if entry and datetime.utcnow() - entry["ts"] < NASA_CACHE_TTL:
            return entry["data"]
    return None


def _set_cached_nasa(lat: float, lon: float, data: Any):
    key = _nasa_cache_key(lat, lon)
    with _nasa_cache_lock:
        _nasa_cache[key] = {"data": data, "ts": datetime.utcnow()}


def _make_job_id(request: PredictRequest) -> str:
    """Deterministic job id so identical requests reuse results."""
    raw = f"{request.latitude:.4f}:{request.longitude:.4f}:{request.prediction_type}:{request.start}:{request.end}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _run_job(job_id: str, request: PredictRequest):
    """Executed in a background thread."""
    try:
        # 1. NASA data (cached)
        nasa_payload = _get_cached_nasa(request.latitude, request.longitude)
        if nasa_payload is None:
            nasa_payload = fetch_nasa_data(
                latitude=request.latitude,
                longitude=request.longitude,
            )
            _set_cached_nasa(request.latitude, request.longitude, nasa_payload)

        # 2. Model inference
        prediction = run_iterative_forecast(
            nasa_payload=nasa_payload,
            prediction_type=request.prediction_type,
            start=request.start,
            end=request.end,
        )

        result = PredictResponse(
            status="success",
            prediction_type=request.prediction_type,
            start=request.start,
            end=request.end,
            prediction=prediction,
            message="NASA data fetched, model prediction generated, and results aggregated successfully",
        ).model_dump(mode="json")

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result

    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "solar-prediction-api", "version": "2.0.0"}


@app.post("/predict/async", status_code=202)
def predict_async(request: PredictRequest, background_tasks: BackgroundTasks):
    """
    Accepts the request and immediately returns a job_id.
    The actual computation runs in a background thread.
    """
    job_id = _make_job_id(request)

    with _jobs_lock:
        existing = _jobs.get(job_id)
        # Reuse a completed job if it's less than 1 hour old
        if existing:
            age = datetime.utcnow() - existing["created_at"]
            if existing["status"] in ("pending", "running"):
                return {"job_id": job_id, "status": existing["status"]}
            if existing["status"] == "done" and age < timedelta(hours=1):
                return {"job_id": job_id, "status": "done"}

        _jobs[job_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": datetime.utcnow(),
        }

    background_tasks.add_task(_run_job, job_id, request)
    return {"job_id": job_id, "status": "pending"}


@app.get("/predict/status/{job_id}")
def predict_status(job_id: str):
    """Poll this endpoint until status == 'done' or 'error'."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": job["status"], "error": job.get("error")}


@app.get("/predict/result/{job_id}")
def predict_result(job_id: str):
    """Fetch the full prediction result once status == 'done'."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is not done yet: {job['status']}")
    return job["result"]


# ─── Legacy sync endpoint (kept for compatibility) ───────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    nasa_payload = _get_cached_nasa(request.latitude, request.longitude)
    if nasa_payload is None:
        nasa_payload = fetch_nasa_data(latitude=request.latitude, longitude=request.longitude)
        _set_cached_nasa(request.latitude, request.longitude, nasa_payload)

    prediction = run_iterative_forecast(
        nasa_payload=nasa_payload,
        prediction_type=request.prediction_type,
        start=request.start,
        end=request.end,
    )
    return PredictResponse(
        status="success",
        prediction_type=request.prediction_type,
        start=request.start,
        end=request.end,
        prediction=prediction,
        message="Prediction completed successfully",
    )
