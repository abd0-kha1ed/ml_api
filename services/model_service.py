from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException


MODEL_FILE = Path("models") / "solar_forecast_bundle.joblib"


FEATURES_GHI = [
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "T2M_clim",
    "WS2M_clim",
    "GHI_lag1",
    "GHI_lag7",
    "GHI_lag30",
    "GHI_roll7",
    "GHI_roll30",
]

FEATURES_T2M = [
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "T2M_clim",
    "WS2M_clim",
    "T2M_lag1",
    "T2M_lag7",
    "T2M_lag30",
    "T2M_roll7",
    "T2M_roll30",
]

FEATURES_WS2M = [
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "T2M_clim",
    "WS2M_clim",
    "WS2M_lag1",
    "WS2M_lag7",
    "WS2M_lag30",
    "WS2M_roll7",
    "WS2M_roll30",
]


@lru_cache(maxsize=1)
def load_model_bundle() -> Dict[str, Any]:
    if not MODEL_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model bundle not found: {MODEL_FILE}",
        )

    bundle = joblib.load(MODEL_FILE)

    model_ghi = bundle.get("model_GHI") or bundle.get("model")
    model_t2m = bundle.get("model_T2M")
    model_ws2m = bundle.get("model_WS2M")

    if model_ghi is None or model_t2m is None or model_ws2m is None:
        raise HTTPException(
            status_code=500,
            detail="Model bundle must contain model_GHI/model, model_T2M, and model_WS2M.",
        )

    return {
        "model_GHI": model_ghi,
        "model_T2M": model_t2m,
        "model_WS2M": model_ws2m,
    }


def nasa_json_to_dataframe(nasa_payload: Dict[str, Any]) -> pd.DataFrame:
    raw_parameters = nasa_payload["properties"]["parameter"]
    coordinates = nasa_payload["geometry"]["coordinates"]

    df = pd.DataFrame(raw_parameters)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df["longitude"] = coordinates[0]
    df["latitude"] = coordinates[1]
    df["elevation"] = coordinates[2] if len(coordinates) > 2 else 0.0

    required = ["ALLSKY_SFC_SW_DWN", "T2M", "WS2M"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"NASA response missing required parameters: {missing}",
        )

    df[required] = df[required].apply(pd.to_numeric, errors="coerce")
    df[required] = df[required].interpolate(method="linear").ffill().bfill()

    return df.sort_values("Date").reset_index(drop=True)


def prepare_historical_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    df["day"] = df["Date"].dt.dayofyear
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day"] / 365.25)

    clim = (
        df.groupby(["month", "day"])[["T2M", "WS2M", "ALLSKY_SFC_SW_DWN"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "T2M": "T2M_clim",
                "WS2M": "WS2M_clim",
                "ALLSKY_SFC_SW_DWN": "GHI_clim",
            }
        )
    )

    df = df.merge(clim, on=["month", "day"], how="left")
    df[["T2M_clim", "WS2M_clim", "GHI_clim"]] = (
        df[["T2M_clim", "WS2M_clim", "GHI_clim"]]
        .interpolate(method="linear")
        .ffill()
        .bfill()
    )

    return df, clim


def build_future_frame(
    clim: pd.DataFrame,
    start: date,
    end: date,
    prediction_type: str,
) -> pd.DataFrame:
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    if end_date < start_date:
        raise HTTPException(
            status_code=400,
            detail="end date must be after or equal start date.",
        )

    freq = "3D" if prediction_type == "monthly" else "7D"
    future_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    if future_dates.empty:
        raise HTTPException(
            status_code=400,
            detail="No forecast dates generated for the provided date range.",
        )

    df_future = pd.DataFrame({"Date": future_dates})
    df_future["day"] = df_future["Date"].dt.dayofyear
    df_future["month"] = df_future["Date"].dt.month
    df_future["year"] = df_future["Date"].dt.year

    df_future["month_sin"] = np.sin(2 * np.pi * df_future["month"] / 12)
    df_future["month_cos"] = np.cos(2 * np.pi * df_future["month"] / 12)
    df_future["doy_sin"] = np.sin(2 * np.pi * df_future["day"] / 365.25)
    df_future["doy_cos"] = np.cos(2 * np.pi * df_future["day"] / 365.25)

    df_future = df_future.merge(clim, on=["month", "day"], how="left")
    df_future[["T2M_clim", "WS2M_clim", "GHI_clim"]] = (
        df_future[["T2M_clim", "WS2M_clim", "GHI_clim"]]
        .interpolate(method="linear")
        .ffill()
        .bfill()
    )

    return df_future


def mean_tail(values: list[float], count: int) -> float:
    if not values:
        return 0.0
    return float(np.mean(values[-count:])) if len(values) >= count else float(np.mean(values))


def value_lag(values: list[float], lag: int) -> float:
    if not values:
        return 0.0
    return float(values[-lag]) if len(values) >= lag else float(values[0])


def _predict_one(model: Any, row: Dict[str, float], columns: list[str]) -> float:
    x = pd.DataFrame([{col: row[col] for col in columns}], columns=columns)
    return float(model.predict(x)[0])


def _is_constant(values: list[float], tolerance: float = 1e-6) -> bool:
    if len(values) < 3:
        return False
    return float(np.std(values)) <= tolerance


def run_iterative_forecast(
    nasa_payload: Dict[str, Any],
    prediction_type: Literal["monthly", "yearly"],
    start: date,
    end: date,
) -> Dict[str, Any]:
    bundle = load_model_bundle()

    model_ghi = bundle["model_GHI"]
    model_t2m = bundle["model_T2M"]
    model_ws2m = bundle["model_WS2M"]

    df = nasa_json_to_dataframe(nasa_payload)
    df, clim = prepare_historical_frame(df)
    df_future = build_future_frame(clim, start, end, prediction_type)

    history_ghi = df["ALLSKY_SFC_SW_DWN"].astype(float).tolist()
    history_t2m = df["T2M"].astype(float).tolist()
    history_ws2m = df["WS2M"].astype(float).tolist()

    preds_ghi: list[float] = []
    preds_t2m: list[float] = []
    preds_ws2m: list[float] = []

    for _, row in df_future.iterrows():
        base = {
            "month_sin": float(row["month_sin"]),
            "month_cos": float(row["month_cos"]),
            "doy_sin": float(row["doy_sin"]),
            "doy_cos": float(row["doy_cos"]),
            "T2M_clim": float(row["T2M_clim"]),
            "WS2M_clim": float(row["WS2M_clim"]),
        }

        ghi_features = {
            **base,
            "GHI_lag1": value_lag(history_ghi, 1),
            "GHI_lag7": value_lag(history_ghi, 7),
            "GHI_lag30": value_lag(history_ghi, 30),
            "GHI_roll7": mean_tail(history_ghi, 7),
            "GHI_roll30": mean_tail(history_ghi, 30),
        }

        t2m_features = {
            **base,
            "T2M_lag1": value_lag(history_t2m, 1),
            "T2M_lag7": value_lag(history_t2m, 7),
            "T2M_lag30": value_lag(history_t2m, 30),
            "T2M_roll7": mean_tail(history_t2m, 7),
            "T2M_roll30": mean_tail(history_t2m, 30),
        }

        ws2m_features = {
            **base,
            "WS2M_lag1": value_lag(history_ws2m, 1),
            "WS2M_lag7": value_lag(history_ws2m, 7),
            "WS2M_lag30": value_lag(history_ws2m, 30),
            "WS2M_roll7": mean_tail(history_ws2m, 7),
            "WS2M_roll30": mean_tail(history_ws2m, 30),
        }

        ghi = max(_predict_one(model_ghi, ghi_features, FEATURES_GHI), 0.0)
        t2m = _predict_one(model_t2m, t2m_features, FEATURES_T2M)
        ws2m = max(_predict_one(model_ws2m, ws2m_features, FEATURES_WS2M), 0.0)

        preds_ghi.append(ghi)
        preds_t2m.append(t2m)
        preds_ws2m.append(ws2m)

        history_ghi.append(ghi)
        history_t2m.append(t2m)
        history_ws2m.append(ws2m)

    df_future["Predicted_GHI"] = preds_ghi
    df_future["Predicted_T2M"] = preds_t2m
    df_future["Predicted_WS2M"] = preds_ws2m

    result = aggregate_forecast(df_future, nasa_payload, prediction_type)

    diagnostics = {
        "constant_warning": {
            "ALLSKY_SFC_SW_DWN": _is_constant(preds_ghi),
            "T2M": _is_constant(preds_t2m),
            "WS2M": _is_constant(preds_ws2m),
        },
        "std": {
            "ALLSKY_SFC_SW_DWN": round(float(np.std(preds_ghi)), 6),
            "T2M": round(float(np.std(preds_t2m)), 6),
            "WS2M": round(float(np.std(preds_ws2m)), 6),
        },
    }

    result["header"]["diagnostics"] = diagnostics
    return result


def aggregate_forecast(
    df_future: pd.DataFrame,
    nasa_src: Dict[str, Any],
    prediction_type: str,
) -> Dict[str, Any]:
    df = df_future.copy()

    if prediction_type == "monthly":
        df["period"] = df["Date"].dt.strftime("%Y-%m")
    else:
        df["period"] = df["Date"].dt.strftime("%Y")

    grouped = (
        df.groupby("period")[["Predicted_GHI", "Predicted_T2M", "Predicted_WS2M"]]
        .mean()
        .round(4)
    )

    def to_dict(col: str) -> Dict[str, float]:
        return {
            str(period): float(value)
            for period, value in grouped[col].to_dict().items()
        }

    return {
        "type": "Feature",
        "geometry": nasa_src.get("geometry"),
        "header": {
            "title": "NASA/POWER Forecast Aggregated By XGBoost",
            "prediction_type": prediction_type,
            "aggregation": "average",
            "start": df["Date"].min().strftime("%Y-%m-%d"),
            "end": df["Date"].max().strftime("%Y-%m-%d"),
            "model": "XGBoost Iterative Forecast",
        },
        "properties": {
            "parameter": {
                "ALLSKY_SFC_SW_DWN": to_dict("Predicted_GHI"),
                "T2M": to_dict("Predicted_T2M"),
                "WS2M": to_dict("Predicted_WS2M"),
            }
        },
    }