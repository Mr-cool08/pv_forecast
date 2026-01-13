import sys
import subprocess
from pathlib import Path
from datetime import date, datetime

import pandas as pd
import requests
from flask import Flask, render_template, redirect, url_for, flash

from config import CFG

BASE_DIR = Path(__file__).resolve().parent
PRED_DIR = BASE_DIR / CFG.predictions_dir

# Elpris-API (spotpris per elområde)
ELPRICE_API_BASE = "https://www.elprisetjustnu.se/api/v1/prices"

app = Flask(__name__)
app.secret_key = "pv_forecast_secret"  # byt gärna till något eget


def find_latest_prediction_file() -> Path | None:
    if not PRED_DIR.exists():
        return None
    files = list(PRED_DIR.glob("prediction_TOTAL_*.csv"))
    if not files:
        return None

    # Filnamn: prediction_TOTAL_YYYY-MM-DD.csv
    def key(p: Path):
        stem = p.stem.replace("prediction_TOTAL_", "")
        try:
            return date.fromisoformat(stem)
        except Exception:
            return date.fromtimestamp(p.stat().st_mtime)

    return sorted(files, key=key)[-1]


def read_prediction(path: Path) -> dict:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Predictionsfilen är tom.")
    row = df.iloc[0].to_dict()
    row["__file__"] = path.name
    return row


def fetch_forecast_for_day(target_day_iso: str) -> dict:
    """
    Hämtar daily forecast för exakt target date.
    Returnerar dict med date + alla CFG.weather_daily_vars.
    """
    params = {
        "latitude": CFG.latitude,
        "longitude": CFG.longitude,
        "daily": ",".join(CFG.weather_daily_vars),
        "timezone": CFG.timezone,
        "start_date": target_day_iso,
        "end_date": target_day_iso,
    }
    r = requests.get(CFG.open_meteo_forecast_url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    daily = j.get("daily", {})
    if not daily or "time" not in daily:
        raise ValueError(f"Oväntat forecastsvar: {list(j.keys())}")

    out = {"date": daily["time"][0]}
    for var in CFG.weather_daily_vars:
        v = daily.get(var)
        out[var] = (v[0] if isinstance(v, list) and len(v) > 0 else None)
    return out


def fetch_elprices_for_day(target_day_iso: str) -> dict:
    """
    Hämtar spotpriser för elområdet som gäller för din plats (default SE2).
    Returnerar summary + serie (24 tim eller 96 kvart).
    Priserna från API:t är utan moms/skatt/påslag.
    """
    area = getattr(CFG, "elprice_area", "SE2")  # Hudiksvall brukar vara SE2
    d = date.fromisoformat(target_day_iso)
    url = f"{ELPRICE_API_BASE}/{d.year}/{d.strftime('%m-%d')}_{area}.json"

    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        raise ValueError(
            f"Inga elpriser hittades för {target_day_iso} ({area}). "
            "API:t brukar ha idag/imorgon och historik t.o.m. 2022-11-01."
        )
    r.raise_for_status()
    series = r.json()

    if not isinstance(series, list) or len(series) == 0:
        raise ValueError("Oväntat svar från elpris-API (tom lista).")

    # plocka SEK_per_kWh och räkna statistik
    prices_sek = []
    for x in series:
        v = x.get("SEK_per_kWh")
        if v is None:
            continue
        try:
            prices_sek.append(float(v))
        except Exception:
            pass

    if not prices_sek:
        raise ValueError("Elpris-API svarade men inga giltiga priser kunde tolkas.")

    # intervallets längd (15 eller 60 min) om möjligt
    interval_minutes = None
    try:
        t0 = datetime.fromisoformat(series[0]["time_start"])
        t1 = datetime.fromisoformat(series[0]["time_end"])
        interval_minutes = int((t1 - t0).total_seconds() // 60)
    except Exception:
        interval_minutes = None

    avg_sek = sum(prices_sek) / len(prices_sek)
    min_sek = min(prices_sek)
    max_sek = max(prices_sek)

    return {
        "date": target_day_iso,
        "area": area,
        "interval_minutes": interval_minutes,
        "count": len(prices_sek),
        "avg_ore": avg_sek * 100.0,
        "min_ore": min_sek * 100.0,
        "max_ore": max_sek * 100.0,
        "series": series,
        "source": "elprisetjustnu.se",
        "url": url,
    }


def run_predict_script() -> tuple[bool, str]:
    """
    Kör din predictor och returnerar (ok, loggtext).
    """
    cmd = [sys.executable, "-m", "scripts.predict_tomorrow"]
    p = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return (p.returncode == 0, out.strip())


@app.route("/", methods=["GET"])
def index():
    pred_file = find_latest_prediction_file()
    prediction = None
    weather = None
    elprices = None
    error = None

    try:
        if pred_file is None:
            raise FileNotFoundError(
                f"Ingen predictionsfil hittades i {PRED_DIR}. Kör först predict eller tryck Uppdatera prognos."
            )
        prediction = read_prediction(pred_file)

        # vi hämtar väder + elpris för samma datum som prediktionen avser
        target_day_iso = str(prediction.get("date"))
        weather = fetch_forecast_for_day(target_day_iso)
        elprices = fetch_elprices_for_day(target_day_iso)

    except Exception as e:
        error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        weather=weather,
        elprices=elprices,
        error=error,
        weather_vars=list(CFG.weather_daily_vars),
        lat=CFG.latitude,
        lon=CFG.longitude,
        tz=CFG.timezone,
    )


@app.route("/refresh", methods=["POST"])
def refresh():
    ok, logtxt = run_predict_script()
    if ok:
        flash("✅ Prognosen uppdaterades.", "success")
    else:
        flash("❌ Kunde inte uppdatera prognosen. Se logg nedan.", "error")
        flash(logtxt[-3000:], "log")  # sista delen av loggen
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Kör: python app.py
    app.run(host="127.0.0.1", port=5000, debug=True)
