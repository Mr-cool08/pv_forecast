import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

from config import CFG
from scripts._common import ensure_dir


def build_features_frame(prod_total: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = prod_total.merge(weather, on="date", how="inner").sort_values("date").copy()

    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek
    df["doy"] = df["date"].dt.dayofyear

    lags = (1, 2, 3)
    for lag in lags:
        df[f"kwh_lag{lag}"] = df["kwh"].shift(lag)

    windows = (7, 14)
    for w in windows:
        df[f"kwh_ma{w}"] = df["kwh"].rolling(window=w, min_periods=max(2, w // 2)).mean().shift(1)

    feature_cols = (
        ["month", "dow", "doy"]
        + [f"kwh_lag{lag}" for lag in lags]
        + [f"kwh_ma{w}" for w in windows]
        + list(CFG.weather_daily_vars)
    )

    df = df.dropna(subset=feature_cols + ["kwh"]).copy()

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols + ["kwh"]).copy()

    return df, feature_cols


def main():
    ensure_dir("plots")

    prod = pd.read_csv("data/production_total.csv", parse_dates=["date"]).sort_values("date")
    weather = pd.read_csv(CFG.weather_historical_path, parse_dates=["date"]).sort_values("date")

    bundle = joblib.load(os.path.join(CFG.models_dir, "TOTAL.joblib"))
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df, _ = build_features_frame(prod, weather)

    X = df[feature_cols].copy()
    y_true = df["kwh"].astype(float).values
    dates = df["date"].values

    # Prediktera historiken med den sparade modellen
    # (använder DMatrix för att undvika device-mismatch-varning)
    dmat = xgb.DMatrix(X)
    y_pred = model.get_booster().predict(dmat)

    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Plot
    plt.figure()
    plt.plot(dates, y_true, label="Faktisk (kWh)")
    plt.plot(dates, y_pred, label="Predikterad (kWh)")
    plt.title(f"Backtest TOTAL – MAE ≈ {mae:.2f} kWh")
    plt.xlabel("Datum")
    plt.ylabel("kWh")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join("plots", "backtest_total.png")
    plt.savefig(out_path, dpi=160)
    print(f"Skrev {out_path}")


if __name__ == "__main__":
    main()
