import pandas as pd
import requests
from config import CFG
from scripts._common import ensure_dir


def main():
    ensure_dir("data")

    # LÄS TOTAL-PRODUKTION (VIKTIGT)
    prod = pd.read_csv("data/production_total.csv", parse_dates=["date"])
    if prod.empty or "date" not in prod.columns:
        raise SystemExit("production_total.csv saknas eller saknar kolumnen 'date'.")

    start = prod["date"].min().date().isoformat()
    end = prod["date"].max().date().isoformat()

    params = {
        "latitude": CFG.latitude,
        "longitude": CFG.longitude,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(CFG.weather_daily_vars),
        "timezone": CFG.timezone,
    }

    r = requests.get(CFG.open_meteo_archive_url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()

    daily = j.get("daily", {})
    if not daily or "time" not in daily:
        raise SystemExit(f"Oväntat svar från Open-Meteo: {j.keys()}")

    w = pd.DataFrame(daily)
    w["date"] = pd.to_datetime(w["time"])
    w = w.drop(columns=["time"]).sort_values("date")

    w.to_csv(CFG.weather_historical_path, index=False)
    print(f"Skrev {CFG.weather_historical_path} för {start} till {end}.")


if __name__ == "__main__":
    main()
