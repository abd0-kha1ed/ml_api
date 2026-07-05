import hashlib
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from fastapi.responses import JSONResponse, Response
from fastapi.responses import HTMLResponse

from fastapi import BackgroundTasks, FastAPI, HTTPException

from schemas.predict_schema import PredictRequest, PredictResponse
from services.model_service import run_iterative_forecast
from services.nasa_service import fetch_nasa_data

app = FastAPI(
    title="Solar Prediction API",
    version="2.1.0",
    description="Async Solar Prediction API with job tracking and progress stages",
)

_jobs: Dict[str, Any] = {}
_jobs_lock = threading.Lock()

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
    raw = (
        f"{request.latitude:.4f}:"
        f"{request.longitude:.4f}:"
        f"{request.prediction_type}:"
        f"{request.start}:"
        f"{request.end}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return

        if status is not None:
            job["status"] = status
        if stage is not None:
            job["stage"] = stage
        if progress is not None:
            job["progress"] = max(0.0, min(float(progress), 1.0))
        if message is not None:
            job["message"] = message
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error

        job["updated_at"] = datetime.utcnow()


def _run_job(job_id: str, request: PredictRequest):
    try:
        _update_job(
            job_id,
            status="running",
            stage="nasa",
            progress=0.20,
            message="Fetching NASA POWER historical data...",
        )

        nasa_payload = _get_cached_nasa(request.latitude, request.longitude)

        if nasa_payload is None:
            nasa_payload = fetch_nasa_data(
                latitude=request.latitude,
                longitude=request.longitude,
            )
            _set_cached_nasa(request.latitude, request.longitude, nasa_payload)
        else:
            _update_job(
                job_id,
                status="running",
                stage="nasa_cached",
                progress=0.35,
                message="Using cached NASA POWER data...",
            )

        _update_job(
            job_id,
            status="running",
            stage="model",
            progress=0.60,
            message="Running XGBoost solar forecast model...",
        )

        prediction = run_iterative_forecast(
            nasa_payload=nasa_payload,
            prediction_type=request.prediction_type,
            start=request.start,
            end=request.end,
        )

        _update_job(
            job_id,
            status="running",
            stage="aggregation",
            progress=0.85,
            message="Aggregating forecast results...",
        )

        result = PredictResponse(
            status="success",
            prediction_type=request.prediction_type,
            start=request.start,
            end=request.end,
            prediction=prediction,
            message=(
                "NASA data fetched, model prediction generated, "
                "and results aggregated successfully"
            ),
        ).model_dump(mode="json")

        _update_job(
            job_id,
            status="done",
            stage="done",
            progress=1.0,
            message="Prediction completed successfully.",
            result=result,
            error=None,
        )

    except Exception as exc:
        _update_job(
            job_id,
            status="error",
            stage="error",
            progress=1.0,
            message="Prediction failed.",
            error=str(exc),
        )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "solar-prediction-api",
        "version": "2.1.0",
    }


@app.post("/predict/async", status_code=202)
def predict_async(request: PredictRequest, background_tasks: BackgroundTasks):
    job_id = _make_job_id(request)

    with _jobs_lock:
        existing = _jobs.get(job_id)

        if existing:
            age = datetime.utcnow() - existing["created_at"]

            if existing["status"] in ("pending", "running"):
                return {
                    "job_id": job_id,
                    "status": existing["status"],
                    "stage": existing.get("stage"),
                    "progress": existing.get("progress", 0.0),
                    "message": existing.get("message"),
                }

            if existing["status"] == "done" and age < timedelta(hours=1):
                return {
                    "job_id": job_id,
                    "status": "done",
                    "stage": "done",
                    "progress": 1.0,
                    "message": "Prediction already completed.",
                }

        now = datetime.utcnow()
        _jobs[job_id] = {
            "status": "pending",
            "stage": "queued",
            "progress": 0.05,
            "message": "Prediction job queued.",
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }

    background_tasks.add_task(_run_job, job_id, request)

    return {
        "job_id": job_id,
        "status": "pending",
        "stage": "queued",
        "progress": 0.05,
        "message": "Prediction job queued.",
    }


@app.get("/predict/status/{job_id}")
def predict_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job_id,
        "status": job["status"],
        "stage": job.get("stage"),
        "progress": job.get("progress", 0.0),
        "message": job.get("message"),
        "error": job.get("error"),
    }


@app.get("/predict/result/{job_id}")
def predict_result(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not done yet: {job['status']}",
        )

    return job["result"]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    nasa_payload = _get_cached_nasa(request.latitude, request.longitude)

    if nasa_payload is None:
        nasa_payload = fetch_nasa_data(
            latitude=request.latitude,
            longitude=request.longitude,
        )
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

@app.get("/.well-known/assetlinks.json")
def android_assetlinks():
    return JSONResponse(
        content=[
            {
                "relation": ["delegate_permission/common.handle_all_urls"],
                "target": {
                    "namespace": "android_app",
                    "package_name": "com.solix.app",
                    "sha256_cert_fingerprints": [
        "81:B2:E9:A2:74:E2:B9:E3:67:53:15:99:0A:7D:35:8E:E9:1B:F5:DF:21:6A:88:33:91:F3:28:C6:1F:0E:10:27",
        "5D:80:A4:6F:E2:41:B4:1D:84:DB:D5:44:37:43:B3:35:E4:9E:4E:40:36:36:F1:93:3C:3B:B1:09:86:7D:75:22",
        "A5:8F:6D:EE:92:28:60:5D:6F:E8:8C:62:94:9D:18:06:B5:D9:50:4A:5B:2B:83:71:2E:A9:40:59:94:52:86:AC"
      ],
                },
            }
        ]
    )


@app.get("/.well-known/apple-app-site-association")
def apple_app_site_association():
    return JSONResponse(
        content={
            "applinks": {
                "apps": [],
                "details": [
                    {
                        "appID": "com.solix.app",
                        "paths": ["*"],
                    }
                ],
            }
        },
        media_type="application/json",
    )

@app.get("/privacy-policy", response_class=HTMLResponse)
def privacy_policy():
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Privacy Policy - Solix</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          max-width: 900px;
          margin: 40px auto;
          padding: 20px;
          line-height: 1.8;
        }
      </style>
    </head>
    <body>

      <h1>Privacy Policy for Solix</h1>

      <p>Last Updated: June 2026</p>

      <p>
      Solix respects your privacy and is committed to protecting your personal information.
      This Privacy Policy explains how we collect, use, store and protect information
      when you use the Solix mobile application.
      </p>

      <h2>Information We Collect</h2>
      <ul>
        <li>Name</li>
        <li>Email Address</li>
        <li>Authentication information through Google Sign-In</li>
        <li>Information required to provide application services</li>
      </ul>

      <h2>How We Use Information</h2>
      <ul>
        <li>Create and manage user accounts</li>
        <li>Authenticate users</li>
        <li>Provide application functionality</li>
        <li>Improve security and performance</li>
      </ul>

      <h2>Third Party Services</h2>
      <p>
      Solix may use trusted third-party services including Google Sign-In
      and Supabase to provide authentication and data storage.
      </p>

      <h2>Data Storage and Security</h2>
      <p>
      User information is stored securely and protected using industry-standard
      security measures.
      </p>

      <h2>Data Sharing</h2>
      <p>
      Solix does not sell personal information.
      Data is only shared when necessary to provide application functionality
      or comply with legal obligations.
      </p>

      <h2>Data Retention</h2>
      <p>
      We retain user information while the account remains active
      and as required to provide our services.
      </p>

      <h2>Account Deletion</h2>
      <p>
      Users may delete their account directly from the application settings.
      </p>

      <h2>Children's Privacy</h2>
      <p>
      Solix is not intended for children under 13 years of age.
      </p>

      <h2>Changes to This Policy</h2>
      <p>
      We may update this Privacy Policy periodically.
      </p>

      <h2>Contact Us</h2>
      <p>
      Email: abdelrahmankhaleddev@gmail.com
      </p>

    </body>
    </html>
    """


@app.get("/delete-account", response_class=HTMLResponse)
def delete_account_page():
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Delete Account - Solix</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          max-width: 900px;
          margin: 40px auto;
          padding: 20px;
          line-height: 1.8;
          color: #111827;
        }
        h1, h2 {
          color: #0f172a;
        }
        a {
          color: #2563eb;
        }
      </style>
    </head>
    <body>

      <h1>Delete Account - Solix</h1>

      <p>
        Solix allows users to permanently delete their account and associated
        personal information directly from within the mobile application.
      </p>

      <h2>How to Delete Your Account</h2>

      <ol>
        <li>Open the Solix mobile application.</li>
        <li>Sign in to your account.</li>
        <li>Navigate to <strong>Settings</strong>.</li>
        <li>Select <strong>Delete Account</strong>.</li>
        <li>Confirm your deletion request.</li>
      </ol>

      <h2>What Data Will Be Deleted</h2>

      <p>
        When you delete your account, the following information will be permanently removed:
      </p>

      <ul>
        <li>Your account profile information.</li>
        <li>Your authentication credentials associated with Solix.</li>
        <li>Your personal information stored within the application.</li>
      </ul>

      <h2>Data Retention</h2>

      <p>
        Some information may be retained temporarily when required for legal,
        security, fraud prevention, or regulatory compliance purposes.
      </p>

      <h2>Deletion Processing Time</h2>

      <p>
        Account deletion requests are typically processed immediately or within
        a reasonable period required to complete data removal from our systems.
      </p>

      <h2>Need Assistance?</h2>

      <p>
        If you experience any issues deleting your account, please contact:
      </p>

      <p>
        <strong>Email:</strong>
        <a href="mailto:abdelrahmankhaleddev@gmail.com">
          abdelrahmankhaleddev@gmail.com
        </a>
      </p>

      <h2>Privacy Policy</h2>

      <p>
        For more information about how we collect and process personal data,
        please review our Privacy Policy:
      </p>

      <p>
        <a href="/privacy-policy">View Privacy Policy</a>
      </p>

    </body>
    </html>
    """