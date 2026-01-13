from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Masbo 124, Hudiksvall (ungefärlig koordinat)
    latitude: float = 61.727
    longitude: float = 17.112
    timezone: str = "Europe/Stockholm"

    # Mappstruktur
    production_raw_dir: str = "data/production_raw"
    production_long_path: str = "data/production_long.csv"
    weather_historical_path: str = "data/weather_historical.csv"
    models_dir: str = "models"
    predictions_dir: str = "predictions"

    # Open-Meteo endpoints
    # Historical (archive): dagliga historiska värden
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    # Forecast: prognos (dagligt)
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"

    # Dagliga vädervariabler (daily=...)
    # shortwave_radiation_sum är ofta starkast signal för PV.
    weather_daily_vars: tuple[str, ...] = (
        "shortwave_radiation_sum",
        "cloudcover_mean",
        "temperature_2m_mean",
    )

    # Features för modellen (lag/rolling)
    lags: tuple[int, ...] = (1, 2, 3)
    rolling_windows: tuple[int, ...] = (7, 14)

CFG = Config()
