import os
import logging
import argparse
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


# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predict")


def parse_args():
    p = argparse.ArgumentParser(description="Predict TOTAL kWh för valfritt datum.")
    p.add_argument(
        "--date",
        default=None,
        help="Datum att prediktera för (YYYY-MM-DD). Ex: 2026-01-15",
    )
    p.add_argument(
        "--days",
        type=int,
        default=None,
        help="Antal dagar från idag i CFG.timezone (0=today, 1=imorgon, 2=osv).",
    )
    return p.parse_args()


def pick_target_date(tz: str, date_str: str | None, days: int | None):
    today = datetime.now(ZoneInfo(tz)).date()

    if date_str and days is not None:
        raise SystemExit("Välj antingen --date eller --days (inte båda).")

    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit("Ogiltigt --date format. Använd YYYY-MM-DD, t.ex. 2026-01-15.")

    if days is not None:
        return today + timedelta(days=int(days))

    # default: imorgon
    return today + timedelta(days=1)


def fetch_archive_daily(target_date_iso: str):
    """Historiska daily-värden via Open-Meteo Archive API."""
    url = getattr(CFG, "open_meteo_archive_url", "https://archive-api.open-meteo.com/v1/archive")
    params = {
        "latitude": CFG.latitude,
        "longitude": CFG.longitude,
        "daily": ",".join(CFG.weather_daily_vars),
        "timezone": CFG.timezone,
        "start_date": target_date_iso,
        "end_date": target_date_iso,
    }

    r = requests.get(url, params=params, timeout=60)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        log.error(f"Open-Meteo archive svar: {r.text}")
        raise

    j = r.json()
    daily = j.get("daily", {})
    if not daily or "time" not in daily or not daily["time"]:
        raise SystemExit(f"Oväntat archive-svar: {list(j.keys())}")

    out = {}
    for var in CFG.weather_daily_vars:
        v = daily.get(var, None)
        out[var] = v[0] if isinstance(v, list) and len(v) > 0 else None

    out["date"] = pd.to_datetime(daily["time"][0])
    return out


def fetch_forecast_daily(target_date_iso: str):
    """Hämtar daily för en specifik dag.

    Viktigt:
    - Forecast API klarar bara ett begränsat antal dagar framåt (oftast 16).
    - Genom att fråga med past_days + forecast_days får vi en lista av datum och kan plocka ut exakt target-dag.
    - Om datumet är för långt bak (utanför past_days) använder vi Archive API.
    """
    tz = ZoneInfo(CFG.timezone)
    today = datetime.now(tz).date()
    target = datetime.strptime(target_date_iso, "%Y-%m-%d").date()
    delta = (target - today).days

    max_past = int(getattr(CFG, "past_days", 92))
    max_forecast_days = int(getattr(CFG, "forecast_days", 16))
    max_future = max_forecast_days - 1  # idag ingår, t.ex. 0..15 om 16 dagar

    # För långt fram i tiden => tydligt meddelande
    if delta > max_future:
        raise SystemExit(
            f"Datum {target_date_iso} ligger {delta} dagar framåt. "
            f"Open-Meteo forecast klarar max {max_future} dagar framåt. "
            f"Välj ett närmare datum (t.ex. --days 0..{max_future})."
        )

    # För långt bak => archive
    if delta < -max_past:
        log.info(f"{target_date_iso} är äldre än {max_past} dagar bakåt => använder archive API")
        return fetch_archive_daily(target_date_iso)

    # Inom spannet: forecast med fönster
    params = {
        "latitude": CFG.latitude,
        "longitude": CFG.longitude,
        "daily": ",".join(CFG.weather_daily_vars),
        "timezone": CFG.timezone,
        "past_days": max_past,
        "forecast_days": max_forecast_days,
    }

    r = requests.get(CFG.open_meteo_forecast_url, params=params, timeout=60)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        log.error(f"Open-Meteo forecast svar: {r.text}")
        raise

    j = r.json()
    daily = j.get("daily", {})
    if not daily or "time" not in daily or not daily["time"]:
        raise SystemExit(f"Oväntat forecastsvar: {list(j.keys())}")

    times = daily["time"]  # lista av YYYY-MM-DD
    try:
        idx = times.index(target_date_iso)
    except ValueError:
        raise SystemExit(
            f"Kunde inte hitta {target_date_iso} i forecast-svaret. "
            f"Första/sista dag i svaret: {times[0]} .. {times[-1]}"
        )

    out = {}
    for var in CFG.weather_daily_vars:
        v = daily.get(var, None)
        out[var] = v[idx] if isinstance(v, list) and len(v) > idx else None

    out["date"] = pd.to_datetime(times[idx])
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
    args = parse_args()
    ensure_dir(CFG.predictions_dir)

    prod = pd.read_csv("data/production_total.csv", parse_dates=["date"]).sort_values("date")
    if prod.empty:
        raise SystemExit("data/production_total.csv är tom.")

    target = pick_target_date(CFG.timezone, args.date, args.days)
    target_iso = target.isoformat()

    log.info(f"Skapar prognos för {target_iso} (timezone={CFG.timezone})")

    # Hämta väder (forecast/arkiv)
    wrow = fetch_forecast_daily(target_iso)
    d = pd.Timestamp(wrow["date"])

    # ==========================
    # Bygg features
    # ==========================
    row = {
        "dow": int(d.dayofweek),
        "doy": int(d.dayofyear),
        "sin_doy": float(np.sin(2 * np.pi * int(d.dayofyear) / 365.25)),
        "cos_doy": float(np.cos(2 * np.pi * int(d.dayofyear) / 365.25)),
        "daylight_h": float(compute_daylight_h(d, CFG.latitude)),
    }

    # Om target ligger i historiken/backtest: använd endast historik fram till dagen innan target.
    # Annars (framtid): använder all känd historik (samma logik, eftersom prod_cut blir allt < target).
    target_ts = pd.Timestamp(target)
    prod_cut = prod[prod["date"] < target_ts].copy()

    # laggar (från senaste tillgängliga produktion innan target)
    lags = (1, 2, 7)
    for lag in lags:
        row[f"kwh_lag{lag}"] = float(prod_cut.iloc[-lag]["kwh"]) if len(prod_cut) >= lag else None

    # rullande medel (matchar train: rolling(...).mean().shift(1) => använd historik t.o.m dagen innan target)
    windows = (7, 14, 28)
    for w in windows:
        minp = max(3, w // 2)
        row[f"kwh_ma{w}"] = float(prod_cut["kwh"].tail(w).mean()) if len(prod_cut) >= minp else None

    # EWM och STD (matchar train: kwh.shift(1).ewm(...) / rolling(...).std())
    for span in (7, 14, 28):
        row[f"kwh_ewm{span}"] = (
            float(prod_cut["kwh"].ewm(span=span, adjust=False).mean().iloc[-1])
            if len(prod_cut) >= 2
            else None
        )
        minp = max(3, span // 2)
        row[f"kwh_std{span}"] = float(prod_cut["kwh"].tail(span).std()) if len(prod_cut) >= minp else None

    # väder
    for var in CFG.weather_daily_vars:
        row[var] = wrow.get(var, None)

    # interaktion (om modellen använder den)
    if "shortwave_radiation_sum" in row and row["shortwave_radiation_sum"] is not None:
        row["rad_x_daylight"] = float(row["shortwave_radiation_sum"]) * float(row["daylight_h"])
    else:
        row["rad_x_daylight"] = 0.0

    # ==========================
    # Prediktera
    # ==========================
    bundle = joblib.load(os.path.join(CFG.models_dir, "TOTAL.joblib"))

    transform = bundle.get("target_transform")
    if transform != "log1p_split":
        raise SystemExit(f"Fel modelltyp i TOTAL.joblib: target_transform={transform}")

    feature_cols = bundle["feature_cols"]
    booster_low = bundle["booster_low"]
    booster_high = bundle["booster_high"]
    thr = float(bundle["radiation_threshold"])

    cal_low = bundle.get("calibration_low")  # {"a":..., "b":...} eller None
    cal_high = bundle.get("calibration_high")
    best_iters = bundle.get("best_iterations")  # {"low":..., "high":...} om finns

    # Bygg X exakt med feature_cols
    X = pd.DataFrame([row]).reindex(columns=feature_cols)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0).astype("float32")

    dmat = xgb.DMatrix(X, feature_names=feature_cols)

    rad = float(row.get("shortwave_radiation_sum", 0.0) or 0.0)
    chosen = "LOW" if rad <= thr else "HIGH"
    booster = booster_low if chosen == "LOW" else booster_high

    log.info(f"Radiation={rad:.3f} | thr={thr:.3f} => väljer {chosen}")
    if isinstance(best_iters, dict):
        log.info(f"Best iters (från threshold-val): low={best_iters.get('low')} high={best_iters.get('high')}")

    pred_log = float(booster.predict(dmat)[0])
    yhat = float(np.expm1(pred_log))
    yhat = max(0.0, yhat)

    # Kalibrering (om finns)
    calibration_used = False
    if (
        isinstance(cal_low, dict)
        and isinstance(cal_high, dict)
        and "a" in cal_low
        and "b" in cal_low
        and "a" in cal_high
        and "b" in cal_high
    ):
        a = float(cal_low["a"]) if chosen == "LOW" else float(cal_high["a"])
        b = float(cal_low["b"]) if chosen == "LOW" else float(cal_high["b"])
        yhat = max(0.0, a * yhat + b)
        calibration_used = True

    out = pd.DataFrame(
        [
            {
                "date": target_iso,
                "target": "TOTAL",
                "kwh_pred": yhat,
                "model_chosen": chosen,
                "shortwave_radiation_sum": rad,
                "threshold": thr,
                "calibration_used": calibration_used,
            }
        ]
    )

    out_path = os.path.join(CFG.predictions_dir, f"prediction_TOTAL_{target_iso}.csv")
    out.to_csv(out_path, index=False, float_format="%.2f")
    log.info(f"Skrev {out_path}")
    print(out)

    # ==========================
    # Graf: senaste 30 dagar före target + prognos
    # ==========================
    last_n = 30
    hist = prod[prod["date"] < target_ts].tail(last_n).copy()

    # Om det inte finns någon historik före target (t.ex. väldigt tidigt backtest), fall tillbaka på senaste 30 rader.
    if hist.empty:
        hist = prod.tail(last_n).copy()

    hist["date"] = pd.to_datetime(hist["date"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hist["date"], hist["kwh"], marker="o", linewidth=1.5, label="Faktisk (senaste 30 dagar)")
    ax.scatter([pd.Timestamp(target)], [yhat], s=140, label=f"Prognos ({chosen})")

    ax.set_title(f"TOTAL – senaste 30 dagar + prognos ({target_iso})")
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
    log.info(f"Skrev {img_path}")


if __name__ == "__main__":
    main()
