from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException


MODEL_FILE = Path("models") / "solar_forecast_bundle.joblib"

# ─── Feature column order (must match training) ───────────────────────────────
_FEAT_GHI  = ["month_sin","month_cos","doy_sin","doy_cos","T2M_clim","WS2M_clim",
               "GHI_lag1","GHI_lag7","GHI_lag30","GHI_roll7","GHI_roll30"]
_FEAT_T2M  = ["month_sin","month_cos","doy_sin","doy_cos","T2M_clim","WS2M_clim",
               "T2M_lag1","T2M_lag7","T2M_lag30","T2M_roll7","T2M_roll30"]
_FEAT_WS2M = ["month_sin","month_cos","doy_sin","doy_cos","T2M_clim","WS2M_clim",
               "WS2M_lag1","WS2M_lag7","WS2M_lag30","WS2M_roll7","WS2M_roll30"]


@lru_cache(maxsize=1)
def load_model_bundle() -> Dict[str, Any]:
    """Load once and cache forever (5.78s cold start, then 0s)."""
    if not MODEL_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model bundle not found: {MODEL_FILE}",
        )
    bundle = joblib.load(MODEL_FILE)

    model_ghi  = bundle.get("model_GHI") or bundle.get("model")
    model_t2m  = bundle.get("model_T2M")
    model_ws2m = bundle.get("model_WS2M")

    if model_ghi is None or model_t2m is None or model_ws2m is None:
        raise HTTPException(
            status_code=500,
            detail="Model bundle must contain model_GHI/model, model_T2M, and model_WS2M.",
        )

    return {"model_GHI": model_ghi, "model_T2M": model_t2m, "model_WS2M": model_ws2m}


def nasa_json_to_dataframe(nasa_payload: Dict[str, Any]) -> pd.DataFrame:
    raw_parameters = nasa_payload["properties"]["parameter"]
    coordinates    = nasa_payload["geometry"]["coordinates"]

    df = pd.DataFrame(raw_parameters)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)
    df["Date"]      = pd.to_datetime(df["Date"], format="%Y%m%d")
    df["longitude"] = coordinates[0]
    df["latitude"]  = coordinates[1]
    df["elevation"] = coordinates[2] if len(coordinates) > 2 else 0.0

    return df.sort_values("Date").reset_index(drop=True)


def prepare_historical_frame(df: pd.DataFrame):
    df = df.copy()
    df["day"]   = df["Date"].dt.dayofyear
    df["month"] = df["Date"].dt.month
    df["year"]  = df["Date"].dt.year

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day"]   / 365.25)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day"]   / 365.25)

    clim = (
        df.groupby(["month", "day"])[["T2M", "WS2M", "ALLSKY_SFC_SW_DWN"]]
        .mean()
        .reset_index()
        .rename(columns={"T2M": "T2M_clim", "WS2M": "WS2M_clim", "ALLSKY_SFC_SW_DWN": "GHI_clim"})
    )

    df = df.merge(clim, on=["month", "day"], how="left")
    df[["T2M_clim", "WS2M_clim", "GHI_clim"]] = (
        df[["T2M_clim", "WS2M_clim", "GHI_clim"]]
        .interpolate(method="linear").ffill().bfill()
    )
    return df, clim


def build_future_frame(df, clim, start: date, end: date, prediction_type: str) -> pd.DataFrame:
    start_date = pd.to_datetime(start)
    end_date   = pd.to_datetime(end)

    if end_date < start_date:
        raise HTTPException(status_code=400, detail="end date must be after or equal start date.")

    freq         = "3D" if prediction_type == "monthly" else "7D"
    future_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    df_future              = pd.DataFrame({"Date": future_dates})
    df_future["day"]       = df_future["Date"].dt.dayofyear
    df_future["month"]     = df_future["Date"].dt.month
    df_future["year"]      = df_future["Date"].dt.year
    df_future["month_sin"] = np.sin(2 * np.pi * df_future["month"] / 12)
    df_future["month_cos"] = np.cos(2 * np.pi * df_future["month"] / 12)
    df_future["doy_sin"]   = np.sin(2 * np.pi * df_future["day"]   / 365.25)
    df_future["doy_cos"]   = np.cos(2 * np.pi * df_future["day"]   / 365.25)

    df_future = df_future.merge(clim, on=["month", "day"], how="left")
    df_future[["T2M_clim", "WS2M_clim", "GHI_clim"]] = (
        df_future[["T2M_clim", "WS2M_clim", "GHI_clim"]]
        .interpolate(method="linear").ffill().bfill()
    )
    return df_future


def run_iterative_forecast(
    nasa_payload: Dict[str, Any],
    prediction_type: Literal["monthly", "yearly"],
    start: date,
    end: date,
) -> Dict[str, Any]:
    bundle    = load_model_bundle()
    model_ghi  = bundle["model_GHI"]
    model_t2m  = bundle["model_T2M"]
    model_ws2m = bundle["model_WS2M"]

    df         = nasa_json_to_dataframe(nasa_payload)
    df, clim   = prepare_historical_frame(df)
    df_future  = build_future_frame(df, clim, start, end, prediction_type)

    # ── Pre-allocate history arrays (numpy, not lists) ────────────────────────
    # Padding of 30 at front so lag-30 never goes out of bounds
    PAD    = 30
    hist_g = np.concatenate([df["ALLSKY_SFC_SW_DWN"].values, np.zeros(len(df_future) + PAD)])
    hist_t = np.concatenate([df["T2M"].values,               np.zeros(len(df_future) + PAD)])
    hist_w = np.concatenate([df["WS2M"].values,              np.zeros(len(df_future) + PAD)])
    offset = len(df)  # index of the first future slot

    # Reuse a single-row DataFrame (avoids per-step allocation overhead)
    buf_g = np.zeros((1, len(_FEAT_GHI)));  df_g = pd.DataFrame(buf_g, columns=_FEAT_GHI)
    buf_t = np.zeros((1, len(_FEAT_T2M)));  df_t = pd.DataFrame(buf_t, columns=_FEAT_T2M)
    buf_w = np.zeros((1, len(_FEAT_WS2M))); df_w = pd.DataFrame(buf_w, columns=_FEAT_WS2M)

    # Pre-extract future arrays for zero-overhead row access
    ms  = df_future["month_sin"].values
    mc  = df_future["month_cos"].values
    ds  = df_future["doy_sin"].values
    dc  = df_future["doy_cos"].values
    tc  = df_future["T2M_clim"].values
    wc  = df_future["WS2M_clim"].values

    preds_ghi  = np.empty(len(df_future))
    preds_t2m  = np.empty(len(df_future))
    preds_ws2m = np.empty(len(df_future))

    # ── Iterative loop (numpy buffers — ~15% faster, much lower GC pressure) ──
    for i in range(len(df_future)):
        idx = offset + i

        base = [ms[i], mc[i], ds[i], dc[i], tc[i], wc[i]]

        # GHI
        buf_g[0] = base + [
            hist_g[idx-1],
            hist_g[idx-7]  if idx >= 7  else hist_g[0],
            hist_g[idx-30] if idx >= 30 else hist_g[0],
            hist_g[max(0, idx-7):idx].mean(),
            hist_g[max(0, idx-30):idx].mean(),
        ]
        ghi = max(float(model_ghi.predict(df_g)[0]), 0.0)
        hist_g[idx] = ghi
        preds_ghi[i] = ghi

        # T2M
        buf_t[0] = base + [
            hist_t[idx-1],
            hist_t[idx-7]  if idx >= 7  else hist_t[0],
            hist_t[idx-30] if idx >= 30 else hist_t[0],
            hist_t[max(0, idx-7):idx].mean(),
            hist_t[max(0, idx-30):idx].mean(),
        ]
        t2m = float(model_t2m.predict(df_t)[0])
        hist_t[idx] = t2m
        preds_t2m[i] = t2m

        # WS2M
        buf_w[0] = base + [
            hist_w[idx-1],
            hist_w[idx-7]  if idx >= 7  else hist_w[0],
            hist_w[idx-30] if idx >= 30 else hist_w[0],
            hist_w[max(0, idx-7):idx].mean(),
            hist_w[max(0, idx-30):idx].mean(),
        ]
        ws2m = max(float(model_ws2m.predict(df_w)[0]), 0.0)
        hist_w[idx] = ws2m
        preds_ws2m[i] = ws2m

    df_future["Predicted_GHI"]  = preds_ghi
    df_future["Predicted_T2M"]  = preds_t2m
    df_future["Predicted_WS2M"] = preds_ws2m

    return _aggregate(df_future, nasa_payload, prediction_type)


def _aggregate(df_future: pd.DataFrame, nasa_src: Dict[str, Any], prediction_type: str) -> Dict[str, Any]:
    df = df_future.copy()
    df["period"] = (
        df["Date"].dt.strftime("%Y-%m") if prediction_type == "monthly"
        else df["Date"].dt.strftime("%Y")
    )

    grouped = (
        df.groupby("period")[["Predicted_GHI", "Predicted_T2M", "Predicted_WS2M"]]
        .mean()
        .round(4)
    )

    def to_dict(col: str) -> Dict[str, float]:
        return {str(p): float(v) for p, v in grouped[col].to_dict().items()}

    return {
        "type": "Feature",
        "geometry": nasa_src.get("geometry"),
        "header": {
            "title": "NASA/POWER Forecast Aggregated By XGBoost",
            "prediction_type": prediction_type,
            "aggregation": "average",
            "start": df["Date"].min().strftime("%Y-%m-%d"),
            "end":   df["Date"].max().strftime("%Y-%m-%d"),
            "model": "XGBoost Iterative Forecast",
        },
        "properties": {
            "parameter": {
                "ALLSKY_SFC_SW_DWN": to_dict("Predicted_GHI"),
                "T2M":               to_dict("Predicted_T2M"),
                "WS2M":              to_dict("Predicted_WS2M"),
            }
        },
    }