import pandas as pd
from config import CFG
from scripts._common import ensure_dir

def main():
    ensure_dir("data")

    prod = pd.read_csv(CFG.production_long_path, parse_dates=["date"])
    if prod.empty:
        raise SystemExit("production_long.csv är tom. Kör build_production_long först.")

    total = prod.groupby("date", as_index=False)["kwh"].sum()
    total = total.sort_values("date")
    out_path = "data/production_total.csv"
    total.to_csv(out_path, index=False)
    print(f"Skrev {out_path} med {len(total)} dagar.")

if __name__ == "__main__":
    main()
