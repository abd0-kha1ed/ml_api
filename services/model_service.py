from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException


MODEL_FILE = Path("models") / "solar_forecast_bundle.joblib"


@lru_cache(maxsize=1)
def load_model_bundle() -> Dict[str, Any]:
    if not MODEL_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model bundle not found: {MODEL_FILE}",
        )

    try:
        return joblib.load(MODEL_FILE)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model bundle: {str(exc)}",
        ) from exc


def nasa_json_to_dataframe(nasa_payload: Dict[str, Any]) -> pd.DataFrame:
    try:
        raw_parameters = nasa_payload["properties"]["parameter"]
        coordinates = nasa_payload["geometry"]["coordinates"]

        longitude = coordinates[0]
        latitude = coordinates[1]
        elevation = coordinates[2] if len(coordinates) > 2 else 0.0

        df = pd.DataFrame(raw_parameters)
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)

        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
        df["longitude"] = longitude
        df["latitude"] = latitude
        df["elevation"] = elevation

        df = df.sort_values("Date").reset_index(drop=True)

        required = ["ALLSKY_SFC_SW_DWN", "T2M", "WS2M"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing NASA parameter: {col}")

        return df

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to convert NASA JSON to DataFrame: {str(exc)}",
        ) from exc


def prepare_historical_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    df["day"] = df["Date"].dt.dayofyear
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day"] / 365.25)

    train_mask = df["Date"] <= "2020-12-31"
    if not train_mask.any():
        train_mask = pd.Series([True] * len(df), index=df.index)

    clim = (
        df[train_mask]
        .groupby(["month", "day"])[["T2M", "WS2M", "ALLSKY_SFC_SW_DWN"]]
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
    df: pd.DataFrame,
    clim: pd.DataFrame,
    start: date,
    end: date,
) -> pd.DataFrame:
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    if end_date < start_date:
        raise HTTPException(
            status_code=400,
            detail="end date must be after or equal start date.",
        )

    last_historical_date = df["Date"].max()
    if start_date <= last_historical_date:
        raise HTTPException(
            status_code=400,
            detail=f"start date must be after last NASA historical date: {last_historical_date.date()}",
        )

    future_dates = pd.date_range(start=start_date, end=end_date, freq="D")

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
    if len(values) >= count:
        return float(np.mean(values[-count:]))
    return float(np.mean(values))


def value_lag(values: list[float], count: int) -> float:
    if len(values) >= count:
        return float(values[-count])
    return float(values[0])


def run_iterative_forecast(
    nasa_payload: Dict[str, Any],
    prediction_type: Literal["monthly", "yearly"],
    start: date,
    end: date,
) -> Dict[str, Any]:
    bundle = load_model_bundle()

    model_ghi = bundle.get("model_GHI") or bundle.get("model")
    model_t2m = bundle.get("model_T2M")
    model_ws2m = bundle.get("model_WS2M")

    if model_ghi is None or model_t2m is None or model_ws2m is None:
        raise HTTPException(
            status_code=500,
            detail="Model bundle must contain model_GHI/model, model_T2M, and model_WS2M.",
        )

    df = nasa_json_to_dataframe(nasa_payload)
    df, clim = prepare_historical_frame(df)
    df_future = build_future_frame(df, clim, start, end)

    history_ghi = df["ALLSKY_SFC_SW_DWN"].astype(float).tolist()
    history_t2m = df["T2M"].astype(float).tolist()
    history_ws2m = df["WS2M"].astype(float).tolist()

    predicted_ghi: list[float] = []
    predicted_t2m: list[float] = []
    predicted_ws2m: list[float] = []

    for _, row in df_future.iterrows():
        base = {
            "month_sin": row["month_sin"],
            "month_cos": row["month_cos"],
            "doy_sin": row["doy_sin"],
            "doy_cos": row["doy_cos"],
            "T2M_clim": row["T2M_clim"],
            "WS2M_clim": row["WS2M_clim"],
        }

        x_ghi = pd.DataFrame(
            [
                {
                    **base,
                    "GHI_lag1": value_lag(history_ghi, 1),
                    "GHI_lag7": value_lag(history_ghi, 7),
                    "GHI_lag30": value_lag(history_ghi, 30),
                    "GHI_roll7": mean_tail(history_ghi, 7),
                    "GHI_roll30": mean_tail(history_ghi, 30),
                }
            ]
        )

        x_t2m = pd.DataFrame(
            [
                {
                    **base,
                    "T2M_lag1": value_lag(history_t2m, 1),
                    "T2M_lag7": value_lag(history_t2m, 7),
                    "T2M_lag30": value_lag(history_t2m, 30),
                    "T2M_roll7": mean_tail(history_t2m, 7),
                    "T2M_roll30": mean_tail(history_t2m, 30),
                }
            ]
        )

        x_ws2m = pd.DataFrame(
            [
                {
                    **base,
                    "WS2M_lag1": value_lag(history_ws2m, 1),
                    "WS2M_lag7": value_lag(history_ws2m, 7),
                    "WS2M_lag30": value_lag(history_ws2m, 30),
                    "WS2M_roll7": mean_tail(history_ws2m, 7),
                    "WS2M_roll30": mean_tail(history_ws2m, 30),
                }
            ]
        )

        pred_ghi = max(float(model_ghi.predict(x_ghi)[0]), 0)
        pred_t2m = float(model_t2m.predict(x_t2m)[0])
        pred_ws2m = max(float(model_ws2m.predict(x_ws2m)[0]), 0)

        predicted_ghi.append(pred_ghi)
        predicted_t2m.append(pred_t2m)
        predicted_ws2m.append(pred_ws2m)

        history_ghi.append(pred_ghi)
        history_t2m.append(pred_t2m)
        history_ws2m.append(pred_ws2m)

    df_future["Predicted_GHI"] = predicted_ghi
    df_future["Predicted_T2M"] = predicted_t2m
    df_future["Predicted_WS2M"] = predicted_ws2m

    return df_to_aggregated_json(df_future, nasa_payload, prediction_type)


def df_to_aggregated_json(
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