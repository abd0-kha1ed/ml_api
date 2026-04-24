def mean_tail(values: list[float], count: int) -> float:
    return float(np.mean(values[-count:])) if len(values) >= count else float(np.mean(values))


def value_lag(values: list[float], lag: int) -> float:
    return float(values[-lag]) if len(values) >= lag else float(values[0])


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

    history_ghi = df["ALLSKY_SFC_SW_DWN"].astype(float).tolist()
    history_t2m = df["T2M"].astype(float).tolist()
    history_ws2m = df["WS2M"].astype(float).tolist()

    preds_ghi, preds_t2m, preds_ws2m = [], [], []

    for _, row in df_future.iterrows():
        base = {
            "month_sin": row["month_sin"],
            "month_cos": row["month_cos"],
            "doy_sin": row["doy_sin"],
            "doy_cos": row["doy_cos"],
            "T2M_clim": row["T2M_clim"],
            "WS2M_clim": row["WS2M_clim"],
        }

        x_ghi = pd.DataFrame([{
            **base,
            "GHI_lag1": value_lag(history_ghi, 1),
            "GHI_lag7": value_lag(history_ghi, 7),
            "GHI_lag30": value_lag(history_ghi, 30),
            "GHI_roll7": mean_tail(history_ghi, 7),
            "GHI_roll30": mean_tail(history_ghi, 30),
        }])

        x_t2m = pd.DataFrame([{
            **base,
            "T2M_lag1": value_lag(history_t2m, 1),
            "T2M_lag7": value_lag(history_t2m, 7),
            "T2M_lag30": value_lag(history_t2m, 30),
            "T2M_roll7": mean_tail(history_t2m, 7),
            "T2M_roll30": mean_tail(history_t2m, 30),
        }])

        x_ws2m = pd.DataFrame([{
            **base,
            "WS2M_lag1": value_lag(history_ws2m, 1),
            "WS2M_lag7": value_lag(history_ws2m, 7),
            "WS2M_lag30": value_lag(history_ws2m, 30),
            "WS2M_roll7": mean_tail(history_ws2m, 7),
            "WS2M_roll30": mean_tail(history_ws2m, 30),
        }])

        ghi = max(float(model_ghi.predict(x_ghi)[0]), 0)
        t2m = float(model_t2m.predict(x_t2m)[0])
        ws2m = max(float(model_ws2m.predict(x_ws2m)[0]), 0)

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