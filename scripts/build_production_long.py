import os
import glob
import pandas as pd
from config import CFG
from scripts._common import ensure_dir, parse_date_series, is_unit_row_firstcol, infer_inverter_columns, inverter_id_from_col

def read_one_csv(path: str) -> pd.DataFrame:
    # Läs med rubrikrad (första raden)
    df = pd.read_csv(path)
    if df.shape[0] == 0:
        return df

    # Ta bort enhetsrad om den finns som första data-rad
    if "Datum och tid" in df.columns and df.shape[0] > 0:
        first = df.iloc[0]["Datum och tid"]
        if is_unit_row_firstcol(str(first)):
            df = df.iloc[1:].copy()

    # Datum
    if "Datum och tid" not in df.columns:
        raise ValueError(f"Saknar kolumn 'Datum och tid' i {path}")
    df["date"] = parse_date_series(df["Datum och tid"])
    df = df.dropna(subset=["date"]).copy()

    # Inverter-kolumner (kWh)
    inv_cols = infer_inverter_columns(df.columns)
    if not inv_cols:
        # Om format skiljer sig: försök hitta kolumner som innehåller 'Energi per växelriktare' och INTE 'per kWp'
        inv_cols = [c for c in df.columns if isinstance(c,str) and "Energi per växelriktare" in c and "per kWp" not in c and "|" in c]

    # Behåll bara date + inv_cols
    keep = ["date"] + inv_cols
    df = df[keep].copy()

    # Konvertera till numeriskt
    for c in inv_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Melt long
    out = df.melt(id_vars=["date"], value_vars=inv_cols, var_name="inverter_col", value_name="kwh")
    out = out.dropna(subset=["kwh"]).copy()
    out["inverter_id"] = out["inverter_col"].map(inverter_id_from_col)
    out = out.drop(columns=["inverter_col"])
    return out

def main():
    ensure_dir("data")
    ensure_dir(CFG.production_raw_dir)

    paths = sorted(glob.glob(os.path.join(CFG.production_raw_dir, "*.csv")))
    if not paths:
        raise SystemExit(f"Inga CSV-filer hittades i {CFG.production_raw_dir}")

    all_rows = []
    for p in paths:
        try:
            all_rows.append(read_one_csv(p))
        except Exception as e:
            print(f"Varning: kunde inte läsa {p}: {e}")

    if not all_rows:
        raise SystemExit("Kunde inte läsa någon fil.")

    prod = pd.concat(all_rows, ignore_index=True)
    prod = prod.groupby(["date","inverter_id"], as_index=False)["kwh"].sum()
    prod = prod.sort_values(["inverter_id","date"])
    prod.to_csv(CFG.production_long_path, index=False)
    print(f"Skrev {CFG.production_long_path} med {len(prod)} rader och {prod['inverter_id'].nunique()} växelriktare.")

if __name__ == "__main__":
    main()
