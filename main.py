from fastapi import FastAPI

from schemas.predict_schema import PredictRequest, PredictResponse
from services.model_service import run_iterative_forecast
from services.nasa_service import fetch_nasa_data

app = FastAPI(
    title="Solar Prediction API",
    version="1.1.0",
    description="Middleware API between Flutter app, NASA POWER, and XGBoost model",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "solar-prediction-api",
        "version": "1.1.0",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    nasa_payload = fetch_nasa_data(
        latitude=request.latitude,
        longitude=request.longitude,
    )

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
        message="NASA data fetched, model prediction generated, and results aggregated successfully",
    )