import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import joblib
import xgboost as xgb

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import CFG
from scripts._common import ensure_dir


def fetch_forecast_daily(target_date_iso: str):
    params = {
        "latitude": CFG.latitude,
        "longitude": CFG.longitude,
        "daily": ",".join(CFG.weather_daily_vars),
        "timezone": CFG.timezone,
        "start_date": target_date_iso,
        "end_date": target_date_iso,
    }
    r = requests.get(CFG.open_meteo_forecast_url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    daily = j.get("daily", {})
    if not daily or "time" not in daily:
        raise SystemExit(f"Oväntat forecastsvar: {list(j.keys())}")

    out = {}
    for var in CFG.weather_daily_vars:
        v = daily.get(var, None)
        out[var] = v[0] if isinstance(v, list) and len(v) > 0 else None

    out["date"] = pd.to_datetime(daily["time"][0])  # samma som target_date_iso
    return out


def compute_daylight_h(d: pd.Timestamp, latitude_deg: float) -> float:
    n = int(d.dayofyear)
    lat = np.deg2rad(latitude_deg)
    decl = np.deg2rad(23.44) * np.sin(np.deg2rad((360 / 365.0) * (n - 81)))
    cos_omega = -np.tan(lat) * np.tan(decl)
    cos_omega = float(np.clip(cos_omega, -1.0, 1.0))
    omega = float(np.arccos(cos_omega))
    return float((2 * omega * 24) / (2 * np.pi))


def main():
    ensure_dir(CFG.predictions_dir)

    prod = pd.read_csv("data/production_total.csv", parse_dates=["date"]).sort_values("date")
    if prod.empty:
        raise SystemExit("data/production_total.csv är tom.")

    # ✅ Alltid: dagens datum i CFG.timezone -> imorgon
    today = datetime.now(ZoneInfo(CFG.timezone)).date()
    target = today + timedelta(days=1)
    target_iso = target.isoformat()

    # Hämta väderprognos för imorgon
    wrow = fetch_forecast_daily(target_iso)
    d = pd.Timestamp(wrow["date"])

    # Features (måste matcha train_models.py)
    row = {
        "dow": int(d.dayofweek),
        "doy": int(d.dayofyear),
        "sin_doy": float(np.sin(2 * np.pi * int(d.dayofyear) / 365.25)),
        "cos_doy": float(np.cos(2 * np.pi * int(d.dayofyear) / 365.25)),
        "daylight_h": compute_daylight_h(d, CFG.latitude),
    }

    # laggar (från senaste tillgängliga produktion)
    lags = (1, 2, 7)
    for lag in lags:
        row[f"kwh_lag{lag}"] = float(prod.iloc[-lag]["kwh"]) if len(prod) >= lag else None

    # rullande medel
    windows = (7, 14, 28)
    for w in windows:
        row[f"kwh_ma{w}"] = float(prod.tail(w)["kwh"].mean()) if len(prod) >= 2 else None

    # väder
    for var in CFG.weather_daily_vars:
        row[var] = wrow.get(var, None)

    # ev interaktion om den finns i feature_cols
    if "shortwave_radiation_sum" in row:
        row["rad_x_daylight"] = float((row.get("shortwave_radiation_sum") or 0.0) * row["daylight_h"])

    bundle = joblib.load(os.path.join(CFG.models_dir, "TOTAL.joblib"))

    transform = bundle.get("target_transform")
    if transform != "log1p_split":
        raise SystemExit(f"Fel modelltyp i TOTAL.joblib: target_transform={transform}")

    feature_cols = bundle["feature_cols"]
    booster_low = bundle["booster_low"]
    booster_high = bundle["booster_high"]
    thr = float(bundle["radiation_threshold"])

    cal_low = bundle.get("calibration_low")   # {"a":..., "b":...} eller None
    cal_high = bundle.get("calibration_high")

    X = pd.DataFrame([row]).reindex(columns=feature_cols)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0).astype("float32")

    dmat = xgb.DMatrix(X, feature_names=feature_cols)

    rad = float(row.get("shortwave_radiation_sum", 0.0) or 0.0)
    chosen = "LOW" if rad <= thr else "HIGH"

    booster = booster_low if chosen == "LOW" else booster_high
    pred_log = float(booster.predict(dmat)[0])
    yhat = float(np.expm1(pred_log))
    yhat = max(0.0, yhat)

    # Kalibrering (om finns)
    if cal_low and cal_high:
        a = float(cal_low["a"]) if chosen == "LOW" else float(cal_high["a"])
        b = float(cal_low["b"]) if chosen == "LOW" else float(cal_high["b"])
        yhat = max(0.0, a * yhat + b)

    out = pd.DataFrame([{
        "date": target_iso,
        "target": "TOTAL",
        "kwh_pred": yhat,
        "model_chosen": chosen,
        "shortwave_radiation_sum": rad,
        "threshold": thr,
        "calibration_used": bool(cal_low and cal_high),
    }])

    out_path = os.path.join(CFG.predictions_dir, f"prediction_TOTAL_{target_iso}.csv")
    out.to_csv(out_path, index=False, float_format="%.2f")  # ✅ alltid två decimaler
    print(f"Skrev {out_path}")
    print(out)

    # ===== Graf: senaste 30 dagar + prognos =====
    last_n = 30
    hist = prod.tail(last_n).copy()
    hist["date"] = pd.to_datetime(hist["date"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hist["date"], hist["kwh"], marker="o", linewidth=1.5, label="Faktisk (senaste 30 dagar)")
    ax.scatter([pd.Timestamp(target)], [yhat], s=140, label=f"Prognos imorgon ({chosen})")

    ax.set_title("TOTAL – senaste 30 dagar + prognos")
    ax.set_xlabel("Datum")
    ax.set_ylabel("kWh")

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlim(
        hist["date"].min() - pd.Timedelta(days=1),
        pd.Timestamp(target) + pd.Timedelta(days=1),
    )

    ax.legend()
    fig.tight_layout()

    img_path = os.path.join(CFG.predictions_dir, f"prediction_TOTAL_{target_iso}.png")
    fig.savefig(img_path, dpi=160)
    plt.close(fig)
    print(f"Skrev {img_path}")


if __name__ == "__main__":
    main()
