# PV day-ahead prognos (per växelriktare) – Masbo 124, Hudiksvall

Det här paketet bygger en enkel men robust "day-ahead"-prognos för solcellsproduktion **per växelriktare**:
- Träning på **faktisk historik** (produktion + historiskt väder).
- Drift: hämtar **väderprognos för imorgon** och predikterar kWh per växelriktare.

## 1) Förberedelser
1. Installera Python 3.10+.
2. Skapa virtuell miljö och installera beroenden:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/mac:
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## 2) Lägg in produktionsfiler
Lägg alla era dagliga CSV-rapporter i:
`data/production_raw/`

Filerna kan vara som din exempel-fil: första raden rubriker, andra raden enhetsrad, därefter data.

## 3) Bygg produktionsdatabas (long-format)
```bash
python -m scripts/build_production_long.py
```
Det skapar:
- `data/production_long.csv` (kolumner: date,inverter_id,kwh)

## 4) Hämta historiskt väder (för träning)
```bash
python -m scripts/fetch_weather_historical.py
```
Skapar:
- `data/weather_historical.csv`

## 5) Träna modeller (en per växelriktare)
```bash
python -m scripts/train_models.py
```
Skapar:
- `models/<inverter_id>.joblib`
- `models/model_report.json`

## 6) Prediktera morgondagen (day-ahead)
```bash
python -m scripts/predict_tomorrow.py
```
Skapar:
- `predictions/prediction_<YYYY-MM-DD>.csv`

## 7) Konfiguration
Ändra inställningar i `config.py`, t.ex. lat/lon, timezone och vilka vädervariabler som används.

## Tips
- Har du fler än 1 år data: ännu bättre.
- Om en växelriktare bytt namn i rapporterna: standardisera `inverter_id` i `scripts/build_production_long.py`.

