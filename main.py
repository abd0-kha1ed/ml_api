from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Solar Prediction API",
    version="0.4.0",
    description="Model Serving Layer for PV site suitability and prediction",
)


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    history_years: int = Field(default=10, ge=1, le=30)
    prediction_horizon_days: int = Field(..., ge=1, le=365)
    model_name: str = Field(default="mock_model")


class FeatureSummary(BaseModel):
    total_common_days: int
    avg_radiation: float
    avg_temperature: float
    avg_wind: float
    min_radiation: float
    max_radiation: float
    min_temperature: float
    max_temperature: float
    min_wind: float
    max_wind: float
    monthly_avg_radiation: Dict[str, float]
    monthly_avg_temperature: Dict[str, float]
    monthly_avg_wind: Dict[str, float]


class FeaturesResponse(BaseModel):
    status: str
    message: Optional[str] = None
    features: FeatureSummary


class PredictionOutput(BaseModel):
    predicted_daily_energy: List[float]
    predicted_monthly_energy: float
    estimated_annual_energy: float
    suitability_score: float


class PredictResponse(BaseModel):
    status: str
    model_used: str
    message: Optional[str] = None
    features: FeatureSummary
    prediction: PredictionOutput


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "solar-prediction-api",
        "version": "0.4.0",
    }


def fetch_nasa_data(latitude: float, longitude: float, years: int = 10) -> Dict[str, Any]:
    current_year = datetime.now().year
    start_year = current_year - years

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS2M",
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": f"{start_year}0101",
        "end": f"{current_year}1231",
        "format": "JSON",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch data from NASA POWER: {str(exc)}",
        ) from exc

    if "properties" not in data or "parameter" not in data["properties"]:
        raise HTTPException(
            status_code=500,
            detail="Unexpected NASA POWER response format.",
        )

    return data


def remove_fill_values(parameter_map: Dict[str, float], fill_value: float = -999) -> Dict[str, float]:
    return {
        key: value
        for key, value in parameter_map.items()
        if value is not None and value != fill_value
    }


def build_common_records(
    radiation: Dict[str, float],
    temperature: Dict[str, float],
    wind: Dict[str, float],
) -> List[Dict[str, float]]:
    common_dates = sorted(set(radiation.keys()) & set(temperature.keys()) & set(wind.keys()))

    return [
        {
            "date": date,
            "radiation": radiation[date],
            "temperature": temperature[date],
            "wind": wind[date],
        }
        for date in common_dates
    ]


def average(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def monthly_average(records: List[Dict[str, float]], field_name: str) -> Dict[str, float]:
    monthly_data: Dict[str, List[float]] = {}

    for record in records:
        month = record["date"][4:6]
        monthly_data.setdefault(month, []).append(record[field_name])

    return {
        month: round(sum(values) / len(values), 4)
        for month, values in sorted(monthly_data.items())
    }


def build_features_from_request(request: PredictRequest) -> FeatureSummary:
    nasa_data = fetch_nasa_data(
        latitude=request.latitude,
        longitude=request.longitude,
        years=request.history_years,
    )

    fill_value = nasa_data.get("header", {}).get("fill_value", -999)
    parameters = nasa_data["properties"]["parameter"]

    radiation = remove_fill_values(parameters.get("ALLSKY_SFC_SW_DWN", {}), fill_value)
    temperature = remove_fill_values(parameters.get("T2M", {}), fill_value)
    wind = remove_fill_values(parameters.get("WS2M", {}), fill_value)

    records = build_common_records(radiation, temperature, wind)

    if not records:
        raise HTTPException(status_code=500, detail="No common valid records found.")

    radiation_values = [record["radiation"] for record in records]
    temperature_values = [record["temperature"] for record in records]
    wind_values = [record["wind"] for record in records]

    return FeatureSummary(
        total_common_days=len(records),
        avg_radiation=average(radiation_values),
        avg_temperature=average(temperature_values),
        avg_wind=average(wind_values),
        min_radiation=round(min(radiation_values), 4),
        max_radiation=round(max(radiation_values), 4),
        min_temperature=round(min(temperature_values), 4),
        max_temperature=round(max(temperature_values), 4),
        min_wind=round(min(wind_values), 4),
        max_wind=round(max(wind_values), 4),
        monthly_avg_radiation=monthly_average(records, "radiation"),
        monthly_avg_temperature=monthly_average(records, "temperature"),
        monthly_avg_wind=monthly_average(records, "wind"),
    )


def run_mock_model(features: FeatureSummary, horizon_days: int) -> PredictionOutput:
    # placeholder logic فقط لحين ربط الموديل الحقيقي
    base_daily_energy = round(features.avg_radiation * 0.75, 4)

    predicted_daily_energy = [base_daily_energy] * min(horizon_days, 30)
    predicted_monthly_energy = round(sum(predicted_daily_energy), 4)
    estimated_annual_energy = round(predicted_monthly_energy * 12, 4)

    suitability_score = round(
        min(
            100,
            (features.avg_radiation * 12)
            - (max(features.avg_temperature - 25, 0) * 0.8)
            + (features.avg_wind * 1.5),
        ),
        2,
    )

    return PredictionOutput(
        predicted_daily_energy=predicted_daily_energy,
        predicted_monthly_energy=predicted_monthly_energy,
        estimated_annual_energy=estimated_annual_energy,
        suitability_score=suitability_score,
    )


@app.post("/features", response_model=FeaturesResponse)
def get_features(request: PredictRequest):
    features = build_features_from_request(request)

    return FeaturesResponse(
        status="success",
        message="Features built successfully from NASA POWER data",
        features=features,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    features = build_features_from_request(request)

    prediction = run_mock_model(features, request.prediction_horizon_days)

    return PredictResponse(
        status="success",
        model_used=request.model_name,
        message="Prediction generated successfully",
        features=features,
        prediction=prediction,
    )