import importlib.util
import os
from pathlib import Path

from config import CFG
from scripts import build_production_long, build_production_total, fetch_weather_historical, train_models


def load_mail_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("solarweb_mail", script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Kunde inte ladda mail-scriptet: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def download_production_csvs():
    mail_script = Path("get mail") / "main.py"
    if not mail_script.exists():
        raise SystemExit("Saknar 'get mail/main.py' för att hämta mailbilagor.")

    os.environ.setdefault("SAVE_DIR", CFG.production_raw_dir)
    module = load_mail_module(mail_script)
    module.main()


def main():
    print("Steg 1/4: Hämtar mail och laddar ner produktionsfiler...")
    download_production_csvs()

    print("Steg 2/4: Bygger production_long.csv...")
    build_production_long.main()

    print("Steg 3/4: Bygger production_total.csv och hämtar historiskt väder...")
    build_production_total.main()
    fetch_weather_historical.main()

    print("Steg 4/4: Tränar modeller...")
    train_models.main()

    print("Klart! Modellerna finns i models/ och rapporten i models/model_report.json.")


if __name__ == "__main__":
    main()
