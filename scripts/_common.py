import os
import re
from typing import Iterable
import pandas as pd

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_date_series(s: pd.Series) -> pd.Series:
    # förväntar dd.MM.yyyy
    return pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")

def is_unit_row_firstcol(val: str) -> bool:
    if not isinstance(val, str):
        return False
    v = val.strip()
    return v.startswith("[") and "yyyy" in v

def infer_inverter_columns(columns: Iterable[str]) -> list[str]:
    cols = []
    for c in columns:
        if isinstance(c, str) and c.startswith("Energi per växelriktare |"):
            cols.append(c)
    return cols

def inverter_id_from_col(col: str) -> str:
    # Ex: "Energi per växelriktare | Symo Advanced 20.0-3-M (1)"
    # -> "Symo Advanced 20.0-3-M (1)"
    parts = col.split("|", 1)
    name = parts[1].strip() if len(parts) == 2 else col.strip()
    # städa bort farliga filnamn-tecken
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    return name
