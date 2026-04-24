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
    return joblib.load(MODEL_FILE)


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

    return df.sort_values("Date").reset_index(drop=True)


def prepare_historical_frame(df: pd.DataFrame):
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

    return df.merge(clim, on=["month", "day"]), clim


def build_future_frame(
    df: pd.DataFrame,
    clim: pd.DataFrame,
    start: date,
    end: date,
    prediction_type: str,
) -> pd.DataFrame:

    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    if end_date < start_date:
        raise HTTPException(400, "Invalid date range")

    # 🚀 أهم سطر (تحسين الأداء)
    freq = "3D" if prediction_type == "monthly" else "7D"

    future_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    df_future = pd.DataFrame({"Date": future_dates})

    df_future["day"] = df_future["Date"].dt.dayofyear
    df_future["month"] = df_future["Date"].dt.month
    df_future["year"] = df_future["Date"].dt.year

    df_future["month_sin"] = np.sin(2 * np.pi * df_future["month"] / 12)
    df_future["month_cos"] = np.cos(2 * np.pi * df_future["month"] / 12)
    df_future["doy_sin"] = np.sin(2 * np.pi * df_future["day"] / 365.25)
    df_future["doy_cos"] = np.cos(2 * np.pi * df_future["day"] / 365.25)

    df_future = df_future.merge(clim, on=["month", "day"], how="left")

    df_future.fillna(method="ffill", inplace=True)

    return df_future


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

    df_future = build_future_frame(df, clim, start, end, prediction_type)

    history_ghi = df["ALLSKY_SFC_SW_DWN"].tolist()
    history_t2m = df["T2M"].tolist()
    history_ws2m = df["WS2M"].tolist()

    preds_ghi, preds_t2m, preds_ws2m = [], [], []

    for _, row in df_future.iterrows():
        x = pd.DataFrame([row])

        ghi = model_ghi.predict(x)[0]
        t2m = model_t2m.predict(x)[0]
        ws2m = model_ws2m.predict(x)[0]

        preds_ghi.append(ghi)
        preds_t2m.append(t2m)
        preds_ws2m.append(ws2m)

        history_ghi.append(ghi)
        history_t2m.append(t2m)
        history_ws2m.append(ws2m)

    df_future["Predicted_GHI"] = preds_ghi
    df_future["Predicted_T2M"] = preds_t2m
    df_future["Predicted_WS2M"] = preds_ws2m

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

    def to_dict(col):
        return grouped[col].to_dict()

    return {
        "type": "Feature",
        "geometry": nasa_src.get("geometry"),
        "header": {
            "prediction_type": prediction_type,
            "aggregation": "average",
        },
        "properties": {
            "parameter": {
                "ALLSKY_SFC_SW_DWN": to_dict("Predicted_GHI"),
                "T2M": to_dict("Predicted_T2M"),
                "WS2M": to_dict("Predicted_WS2M"),
            }
        },
    }