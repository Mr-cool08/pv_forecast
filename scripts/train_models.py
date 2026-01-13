import os
import json
import math
import time
import logging
import subprocess
import argparse

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

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
log = logging.getLogger("train")


# -----------------------
# Safeguards / defaults
# -----------------------
CALIB_MIN_POINTS = int(os.environ.get("CALIB_MIN_POINTS", "20"))
CALIB_A_CLIP = (0.7, 1.3)
CALIB_B_CLIP = (-3.0, 3.0)

DEFAULT_KWH_CLIP = float(os.environ.get("KWH_CLIP", "35"))  # din anläggning max ~30; 35 fångar glitchar


def log_system_info():
    log.info(f"XGBoost version: {xgb.__version__}")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT, timeout=5
        )
        log.info("nvidia-smi -L:")
        for line in out.strip().splitlines():
            log.info(f"  {line}")
    except Exception:
        log.info("nvidia-smi ej tillgänglig (ok om PATH saknas eller ingen NVIDIA).")


# -----------------------
# Small utils
# -----------------------
def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def clip_calibration(a: float, b: float, name: str) -> tuple[float, float]:
    a0, b0 = float(a), float(b)
    a = _clip(a0, CALIB_A_CLIP[0], CALIB_A_CLIP[1])
    b = _clip(b0, CALIB_B_CLIP[0], CALIB_B_CLIP[1])
    if a != a0 or b != b0:
        log.warning(
            f"Kalibrering clip: {name} a {a0:.4f}->{a:.4f} (clip {CALIB_A_CLIP[0]}..{CALIB_A_CLIP[1]}) | "
            f"b {b0:.4f}->{b:.4f} (clip {CALIB_B_CLIP[0]}..{CALIB_B_CLIP[1]})"
        )
    return a, b


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def gate_weight(rad: np.ndarray, thr: float, scale: float) -> np.ndarray:
    """Mjuk gating. scale<=0 => hård gating."""
    rad = np.asarray(rad, dtype=float)
    if scale is None or float(scale) <= 0.0:
        return (rad > float(thr)).astype(float)
    return sigmoid((rad - float(thr)) / float(scale))


def blend_preds(pred_low: np.ndarray, pred_high: np.ndarray, w_high: np.ndarray) -> np.ndarray:
    w = np.asarray(w_high, dtype=float)
    return (1.0 - w) * np.asarray(pred_low, dtype=float) + w * np.asarray(pred_high, dtype=float)


# -----------------------
# Solar geometry helpers
# -----------------------
def day_length_hours(date_ts: pd.Timestamp, latitude_deg: float) -> float:
    n = int(date_ts.dayofyear)
    lat = math.radians(latitude_deg)
    decl = math.radians(23.44) * math.sin(math.radians((360 / 365.0) * (n - 81)))
    cos_omega = -math.tan(lat) * math.tan(decl)
    cos_omega = max(-1.0, min(1.0, cos_omega))
    omega = math.acos(cos_omega)
    return (2 * omega * 24) / (2 * math.pi)


def extraterrestrial_radiation_MJ_m2_day(date_ts: pd.Timestamp, latitude_deg: float) -> float:
    """FAO-56 Ra (MJ/m^2/day). Bra som "clear-sky" proxy."""
    # Constants
    Gsc = 0.0820  # MJ m^-2 min^-1
    n = int(date_ts.dayofyear)
    phi = math.radians(latitude_deg)

    dr = 1.0 + 0.033 * math.cos((2.0 * math.pi / 365.0) * n)
    delta = 0.409 * math.sin((2.0 * math.pi / 365.0) * n - 1.39)

    # sunset hour angle
    cos_ws = -math.tan(phi) * math.tan(delta)
    cos_ws = max(-1.0, min(1.0, cos_ws))
    ws = math.acos(cos_ws)

    Ra = (24.0 * 60.0 / math.pi) * Gsc * dr * (ws * math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.sin(ws))
    return float(max(0.0, Ra))


# -----------------------
# Feature engineering
# -----------------------
def make_features(
    df: pd.DataFrame,
    kwh_clip: float = DEFAULT_KWH_CLIP,
    snow_kwh: float = 0.5,
    snow_clear: float = 0.45,
    snow_window: int = 7,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values("date").copy()

    # Clip target för att ta bort glitchar / outliers (om aktiv)
    if kwh_clip and float(kwh_clip) > 0:
        df["kwh"] = df["kwh"].clip(lower=0.0, upper=float(kwh_clip))

    # Bas
    df["dow"] = df["date"].dt.dayofweek
    df["doy"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df["daylight_h"] = df["date"].apply(lambda d: day_length_hours(pd.Timestamp(d), CFG.latitude))

    # Ra (clear-sky proxy)
    df["ra_MJ"] = df["date"].apply(lambda d: extraterrestrial_radiation_MJ_m2_day(pd.Timestamp(d), CFG.latitude))

    rad_col = "shortwave_radiation_sum"
    has_rad = rad_col in df.columns
    if has_rad:
        # clear-sky index (0..~1) men kan gå över 1 vid data/units; klipp lite
        eps = 1e-6
        df["clear_idx"] = (df[rad_col] / (df["ra_MJ"] + eps)).clip(0.0, 1.5)

        # Lags/MA på radiation (shift(1) för att undvika leakage)
        df["rad_lag1"] = df[rad_col].shift(1)
        df["rad_ma3"] = df[rad_col].shift(1).rolling(3, min_periods=2).mean()
        df["rad_ma7"] = df[rad_col].shift(1).rolling(7, min_periods=4).mean()

        df["clear_idx_lag1"] = df["clear_idx"].shift(1)
        df["clear_idx_ma3"] = df["clear_idx"].shift(1).rolling(3, min_periods=2).mean()

        # Interaktioner
        df["rad_x_daylight"] = df[rad_col] * df["daylight_h"]
        df["rad_x_clear"] = df[rad_col] * df["clear_idx"]

    # Historik på kWh
    lags = (1, 2, 7)
    for lag in lags:
        df[f"kwh_lag{lag}"] = df["kwh"].shift(lag)

    windows = (7, 14, 28)
    for w in windows:
        df[f"kwh_ma{w}"] = df["kwh"].rolling(window=w, min_periods=max(3, w // 2)).mean().shift(1)

    for span in (7, 14, 28):
        df[f"kwh_ewm{span}"] = df["kwh"].shift(1).ewm(span=span, adjust=False).mean()
        df[f"kwh_std{span}"] = df["kwh"].shift(1).rolling(window=span, min_periods=max(3, span // 2)).std()

    # -----------------------
    # Snö-feature (viktigt i Sverige)
    # Byggd på igår: låg produktion TROTS relativt "klar" dag => snö/isk/avbrott
    # -----------------------
    if has_rad and "clear_idx_lag1" in df.columns:
        df["snow_flag"] = (
            (df["kwh_lag1"].fillna(0.0) <= float(snow_kwh))
            & (df["clear_idx_lag1"].fillna(0.0) >= float(snow_clear))
        ).astype(int)
    elif has_rad and "rad_lag1" in df.columns:
        # fallback om clear_idx saknas
        df["snow_flag"] = (
            (df["kwh_lag1"].fillna(0.0) <= float(snow_kwh))
            & (df["rad_lag1"].fillna(0.0) >= float(np.nanpercentile(df[rad_col].values.astype(float), 60)))
        ).astype(int)
    else:
        df["snow_flag"] = 0

    if int(snow_window) > 1:
        df["snow_streak"] = df["snow_flag"].shift(1).rolling(int(snow_window), min_periods=2).sum().fillna(0.0)
    else:
        df["snow_streak"] = df["snow_flag"].shift(1).fillna(0.0)

    # Temp-interaktion om du har temperatur
    temp_candidates = [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "temp_mean",
        "temp_max",
        "temp_min",
    ]
    temp_col = next((c for c in temp_candidates if c in df.columns), None)
    if has_rad and temp_col:
        df["rad_x_temp"] = df[rad_col] * df[temp_col]

    # Featurelist
    base_cols = ["dow", "doy", "sin_doy", "cos_doy", "daylight_h", "ra_MJ"]
    hist_cols = (
        [f"kwh_lag{lag}" for lag in lags]
        + [f"kwh_ma{w}" for w in windows]
        + [f"kwh_ewm{span}" for span in (7, 14, 28)]
        + [f"kwh_std{span}" for span in (7, 14, 28)]
        + ["snow_flag", "snow_streak"]
    )

    rad_feats = []
    if has_rad:
        rad_feats += [
            "clear_idx",
            "rad_lag1",
            "rad_ma3",
            "rad_ma7",
            "clear_idx_lag1",
            "clear_idx_ma3",
            "rad_x_daylight",
            "rad_x_clear",
        ]
        if "rad_x_temp" in df.columns:
            rad_feats.append("rad_x_temp")

    weather_cols = list(CFG.weather_daily_vars)

    feature_cols = base_cols + hist_cols + rad_feats + weather_cols

    # Drop NA
    df = df.dropna(subset=feature_cols + ["kwh"]).copy()
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols + ["kwh"]).copy()

    return df, feature_cols


# -----------------------
# Metrics + calibration
# -----------------------
def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def fit_linear_calibration(
    y_pred: np.ndarray, y_true: np.ndarray, min_points: int = CALIB_MIN_POINTS
) -> tuple[float, float]:
    """Linjär kalibrering: y_true ≈ a*y_pred + b."""
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if len(y_pred) < int(min_points):
        return 1.0, 0.0

    var = float(np.var(y_pred))
    if var < 1e-9:
        return 1.0, 0.0

    cov = float(np.mean((y_pred - y_pred.mean()) * (y_true - y_true.mean())))
    a = cov / var
    b = float(y_true.mean() - a * y_pred.mean())
    return float(a), float(b)


def apply_calibration(y_pred: np.ndarray, a: float, b: float) -> np.ndarray:
    y = float(a) * np.asarray(y_pred, dtype=float) + float(b)
    return np.clip(y, 0.0, None)


# -----------------------
# Custom eval: early stopping på MAE i kWh (inte log-domän)
# -----------------------
def mae_kwh_from_logpred(predt: np.ndarray, dmat: xgb.DMatrix):
    y_true_log = dmat.get_label()
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(predt)
    return "mae_kwh", float(np.mean(np.abs(y_true - y_pred)))


def train_booster_log1p(
    X_tr: pd.DataFrame,
    y_tr_kwh: np.ndarray,
    X_val: pd.DataFrame,
    y_val_kwh: np.ndarray,
    feature_cols: list[str],
    params: dict,
    num_round: int,
    es_rounds: int,
    device: str,
    verbose_eval: int = 0,
):
    y_tr_log = np.log1p(np.asarray(y_tr_kwh, dtype=float))
    y_val_log = np.log1p(np.asarray(y_val_kwh, dtype=float))

    dtrain = xgb.DMatrix(X_tr, label=y_tr_log, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val_log, feature_names=feature_cols)

    params_dev = dict(params)
    params_dev["device"] = device
    params_dev["disable_default_eval_metric"] = 1

    try:
        booster = xgb.train(
            params=params_dev,
            dtrain=dtrain,
            num_boost_round=num_round,
            evals=[(dval, "val")],
            early_stopping_rounds=es_rounds,
            custom_metric=mae_kwh_from_logpred,
            verbose_eval=verbose_eval,
        )
    except TypeError:
        booster = xgb.train(
            params=params_dev,
            dtrain=dtrain,
            num_boost_round=num_round,
            evals=[(dval, "val")],
            early_stopping_rounds=es_rounds,
            feval=mae_kwh_from_logpred,
            verbose_eval=verbose_eval,
        )

    return booster


def retrain_final_booster_log1p(
    X_all: pd.DataFrame,
    y_all_kwh: np.ndarray,
    feature_cols: list[str],
    params: dict,
    device: str,
    num_round: int,
):
    y_all_log = np.log1p(np.asarray(y_all_kwh, dtype=float))
    dtrain = xgb.DMatrix(X_all, label=y_all_log, feature_names=feature_cols)

    params_dev = dict(params)
    params_dev["device"] = device
    params_dev["disable_default_eval_metric"] = 1

    booster = xgb.train(
        params=params_dev,
        dtrain=dtrain,
        num_boost_round=int(num_round),
        evals=[],
        verbose_eval=False,
    )
    return booster


def predict_kwh_from_booster(booster, X: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    d = xgb.DMatrix(X, feature_names=feature_cols)
    pred_log = booster.predict(d)
    pred_kwh = np.expm1(pred_log)
    return np.clip(pred_kwh, 0.0, None)


def predict_kwh_from_ensemble(boosters: list, X: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if not boosters:
        return np.zeros(len(X), dtype=float)
    preds = [predict_kwh_from_booster(b, X, feature_cols) for b in boosters]
    return np.mean(np.stack(preds, axis=0), axis=0)


# -----------------------
# Train helpers
# -----------------------
def train_split_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    rad_col: str,
    thr: float,
    feature_cols: list[str],
    params: dict,
    num_round: int,
    es_rounds: int,
    device: str,
    verbose_eval: int,
    ensemble: int,
    seed0: int = 42,
):
    """Tränar ensemble av LOW/HIGH boosters för given threshold."""
    tr_low = train_df[train_df[rad_col] <= thr]
    tr_high = train_df[train_df[rad_col] > thr]
    va_low = val_df[val_df[rad_col] <= thr]
    va_high = val_df[val_df[rad_col] > thr]

    # Eval-set per del (om för liten, använd hela val)
    X_val_low = (va_low[feature_cols] if len(va_low) >= 10 else val_df[feature_cols])
    y_val_low = (va_low["kwh"].values if len(va_low) >= 10 else val_df["kwh"].values)

    X_val_high = (va_high[feature_cols] if len(va_high) >= 10 else val_df[feature_cols])
    y_val_high = (va_high["kwh"].values if len(va_high) >= 10 else val_df["kwh"].values)

    boosters_low, boosters_high = [], []
    best_iters_low, best_iters_high = [], []

    for i in range(int(max(1, ensemble))):
        seed = int(seed0 + i * 101)
        p = dict(params)
        p["seed"] = seed

        bl = train_booster_log1p(
            tr_low[feature_cols],
            tr_low["kwh"].values,
            X_val_low,
            y_val_low,
            feature_cols,
            p,
            num_round,
            es_rounds,
            device=device,
            verbose_eval=verbose_eval,
        )
        bh = train_booster_log1p(
            tr_high[feature_cols],
            tr_high["kwh"].values,
            X_val_high,
            y_val_high,
            feature_cols,
            p,
            num_round,
            es_rounds,
            device=device,
            verbose_eval=verbose_eval,
        )

        boosters_low.append(bl)
        boosters_high.append(bh)

        bi_low = int(getattr(bl, "best_iteration", 0))
        bi_high = int(getattr(bh, "best_iteration", 0))
        best_iters_low.append(bi_low)
        best_iters_high.append(bi_high)

        try:
            log.info(
                f"  [ens {i+1}/{ensemble}] seed={seed} | LOW best_iter={bi_low} best_score={getattr(bl,'best_score',None)} | "
                f"HIGH best_iter={bi_high} best_score={getattr(bh,'best_score',None)}"
            )
        except Exception:
            pass

    return boosters_low, boosters_high, best_iters_low, best_iters_high


def plot_mae_vs_threshold(rows: list[dict], out_png: str, out_csv: str):
    df = pd.DataFrame(rows).sort_values("thr").reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    if df.empty:
        log.warning("Ingen data att plotta för MAE vs threshold.")
        return

    best_idx = int(df["mae"].idxmin())
    best_thr = float(df.loc[best_idx, "thr"])
    best_mae = float(df.loc[best_idx, "mae"])

    plt.figure(figsize=(10, 6))
    plt.plot(df["thr"].values, df["mae"].values, marker="o")
    plt.title(f"MAE vs threshold (bäst thr={best_thr:.2f}, MAE={best_mae:.3f} kWh)")
    plt.xlabel("Threshold (shortwave_radiation_sum)")
    plt.ylabel("MAE (kWh)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    log.info(f"Skrev graf: {out_png}")
    log.info(f"Skrev data: {out_csv}")


# -----------------------
# Args
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Träna PV-modell (LOW/HIGH) med threshold-scan, soft-gate, ensemble, snö-features, kalibrering och robust sparlogik."
    )
    p.add_argument("--verbose-eval", type=int, default=0, help="XGBoost verbose_eval. 0=tyst, t.ex. 200.")
    p.add_argument("--holdout", type=int, default=60, help="Antal dagar i holdout (val). Default 60.")
    p.add_argument(
        "--calib-days",
        type=int,
        default=35,
        help="Sista N dagar av holdout används för kalibrering (threshold väljs på resterande holdout). Default 35 (bra i Sverige/vinter).",
    )
    p.add_argument(
        "--gate-scale",
        type=float,
        default=0.25,
        help="Soft-gate skala. 0 => hård gating. Default 0.25.",
    )
    p.add_argument("--ensemble", type=int, default=5, help="Antal seeds i ensemble. Default 5.")
    p.add_argument(
        "--kwh-clip",
        type=float,
        default=DEFAULT_KWH_CLIP,
        help="Klipp kWh vid träning (glitch-skydd). 0 eller negativt = av. Default 35.",
    )
    p.add_argument("--snow-kwh", type=float, default=0.5, help="Snöflagga: kwh_lag1 <= denna. Default 0.5.")
    p.add_argument("--snow-clear", type=float, default=0.45, help="Snöflagga: clear_idx_lag1 >= denna. Default 0.45.")
    p.add_argument("--snow-window", type=int, default=7, help="Snow streak fönster. Default 7.")
    return p.parse_args()


# -----------------------
# Main
# -----------------------
def main():
    ensure_dir(CFG.models_dir)
    ensure_dir("plots")

    args = parse_args()
    log_system_info()

    device = "cpu"
    cpu_threads = int(os.environ.get("XGBOOST_NUM_THREADS", os.cpu_count() or 1))
    log.info(f"Vald device: {device} | CPU-trådar: {cpu_threads}")

    t0 = time.time()
    log.info(
        f"Startar träning: scan + soft-gate(scale={args.gate_scale}) + ensemble(n={args.ensemble}) + snö-features + kalibrering + robust save"
    )

    prod = pd.read_csv("data/production_total.csv", parse_dates=["date"]).sort_values("date")
    weather = pd.read_csv(CFG.weather_historical_path, parse_dates=["date"]).sort_values("date")
    data = prod.merge(weather, on="date", how="inner")

    if data.empty:
        raise SystemExit("Ingen överlappning mellan produktion och väder.")

    data, feature_cols = make_features(
        data,
        kwh_clip=float(args.kwh_clip),
        snow_kwh=float(args.snow_kwh),
        snow_clear=float(args.snow_clear),
        snow_window=int(args.snow_window),
    )
    log.info(f"Rader efter features: {len(data)} | Antal features: {len(feature_cols)}")

    # Holdout
    holdout = min(int(args.holdout), max(21, int(0.2 * len(data))))
    train_df = data.iloc[:-holdout].copy()
    val_df = data.iloc[-holdout:].copy()
    log.info(f"Holdout={holdout} | Train={len(train_df)} | Holdout(total)={len(val_df)}")

    # split holdout: select + calib
    calib_days = int(args.calib_days or 0)
    if calib_days > 0 and calib_days < max(10, len(val_df) - 10):
        val_select_df = val_df.iloc[:-calib_days].copy()
        calib_df = val_df.iloc[-calib_days:].copy()
        log.info(f"Holdout split: select={len(val_select_df)} | calib={len(calib_df)} (calib_days={calib_days})")
    else:
        val_select_df = val_df.copy()
        calib_df = val_df.copy()
        if calib_days > 0:
            log.warning("calib-days ignoreras (för stort eller för litet för nuvarande holdout).")

    rad_col = "shortwave_radiation_sum"
    if rad_col not in train_df.columns:
        raise SystemExit(f"Saknar {rad_col} i väderdata.")

    # Params
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 10.0,
        "alpha": 1.0,
        "gamma": 0.5,
        "seed": 42,
        "nthread": cpu_threads,
    }
    num_round = 8000
    es_rounds = 300
    verbose_eval = int(args.verbose_eval)

    # -----------------------
    # Threshold scan (billigt): 1 seed, soft-gate utvärdering
    # -----------------------
    gate_scale = float(args.gate_scale)

    train_rad = train_df[rad_col].values.astype(float)
    coarse_pcts = list(range(20, 81, 5))
    coarse_thrs = [float(np.percentile(train_rad, p)) for p in coarse_pcts]
    log.info(f"Coarse scan percentiler: {coarse_pcts}")

    def eval_thr_single(thr: float) -> float:
        # train 1x low/high
        tr_low = train_df[train_df[rad_col] <= thr]
        tr_high = train_df[train_df[rad_col] > thr]
        va_low = val_select_df[val_select_df[rad_col] <= thr]
        va_high = val_select_df[val_select_df[rad_col] > thr]

        X_val_low = (va_low[feature_cols] if len(va_low) >= 10 else val_select_df[feature_cols])
        y_val_low = (va_low["kwh"].values if len(va_low) >= 10 else val_select_df["kwh"].values)

        X_val_high = (va_high[feature_cols] if len(va_high) >= 10 else val_select_df[feature_cols])
        y_val_high = (va_high["kwh"].values if len(va_high) >= 10 else val_select_df["kwh"].values)

        bl = train_booster_log1p(
            tr_low[feature_cols],
            tr_low["kwh"].values,
            X_val_low,
            y_val_low,
            feature_cols,
            params,
            num_round,
            es_rounds,
            device=device,
            verbose_eval=0,
        )
        bh = train_booster_log1p(
            tr_high[feature_cols],
            tr_high["kwh"].values,
            X_val_high,
            y_val_high,
            feature_cols,
            params,
            num_round,
            es_rounds,
            device=device,
            verbose_eval=0,
        )

        # score on val_select
        X_va = val_select_df[feature_cols]
        rad_va = val_select_df[rad_col].values.astype(float)
        y_true = val_select_df["kwh"].values.astype(float)

        pred_low = predict_kwh_from_booster(bl, X_va, feature_cols)
        pred_high = predict_kwh_from_booster(bh, X_va, feature_cols)
        w = gate_weight(rad_va, thr, gate_scale)
        y_pred = blend_preds(pred_low, pred_high, w)
        return mae(y_true, y_pred)

    best = None
    coarse_rows = []
    for i, (p, thr_) in enumerate(zip(coarse_pcts, coarse_thrs), 1):
        log.info(f"[Coarse {i}/{len(coarse_thrs)}] pct={p} thr={thr_:.2f}")
        t_thr = time.time()
        score = eval_thr_single(thr_)
        coarse_rows.append({"phase": "coarse", "pct": p, "thr": thr_, "mae": score, "seconds": round(time.time() - t_thr, 2)})
        log.info(f"Coarse thr={thr_:.2f} (pct={p}) | MAE={score:.3f} | {time.time() - t_thr:.1f}s")
        if best is None or score < best["mae"]:
            best = {"thr": float(thr_), "mae": float(score)}
            log.info(f"✅ Ny bästa (coarse): thr={thr_:.2f} | MAE={score:.3f}")

    best_thr = best["thr"]
    best_pct = float(np.mean(train_rad <= best_thr) * 100.0)
    fine_pcts = list(range(max(5, int(best_pct) - 8), min(95, int(best_pct) + 8) + 1, 1))
    fine_thrs = [float(np.percentile(train_rad, p)) for p in fine_pcts]
    log.info(f"Fine scan runt pct≈{best_pct:.1f}: percentiler {fine_pcts[0]}..{fine_pcts[-1]}")

    fine_rows = []
    for i, (p, thr_) in enumerate(zip(fine_pcts, fine_thrs), 1):
        log.info(f"[Fine {i}/{len(fine_thrs)}] pct={p} thr={thr_:.2f}")
        t_thr = time.time()
        score = eval_thr_single(thr_)
        fine_rows.append({"phase": "fine", "pct": p, "thr": thr_, "mae": score, "seconds": round(time.time() - t_thr, 2)})
        log.info(f"Fine thr={thr_:.2f} (pct={p}) | MAE={score:.3f} | {time.time() - t_thr:.1f}s")
        if score < best["mae"]:
            best = {"thr": float(thr_), "mae": float(score)}
            log.info(f"✅ Ny bästa (fine): thr={thr_:.2f} | MAE={score:.3f}")

    thr = float(best["thr"])

    # -----------------------
    # Train ensemble on best threshold
    # -----------------------
    log.info(f"Tränar ensemble på vald thr={thr:.2f} (n={args.ensemble})")
    boosters_low, boosters_high, best_iters_low, best_iters_high = train_split_ensemble(
        train_df=train_df,
        val_df=val_select_df,
        rad_col=rad_col,
        thr=thr,
        feature_cols=feature_cols,
        params=params,
        num_round=num_round,
        es_rounds=es_rounds,
        device=device,
        verbose_eval=verbose_eval,
        ensemble=int(args.ensemble),
        seed0=int(params.get("seed", 42)),
    )

    # -----------------------
    # Kalibrering (fit på calib_df) – med soft-gate + ensemble
    # -----------------------
    X_cal = calib_df[feature_cols]
    y_cal_true = calib_df["kwh"].values.astype(float)
    rad_cal = calib_df[rad_col].values.astype(float)

    pred_low_cal = predict_kwh_from_ensemble(boosters_low, X_cal, feature_cols)
    pred_high_cal = predict_kwh_from_ensemble(boosters_high, X_cal, feature_cols)

    w_cal = gate_weight(rad_cal, thr, gate_scale)
    y_cal_pred_raw = blend_preds(pred_low_cal, pred_high_cal, w_cal)

    idx_low = rad_cal <= thr
    idx_high = ~idx_low

    n_low_cal = int(np.sum(idx_low))
    n_high_cal = int(np.sum(idx_high))
    log.info(f"Calib points: LOW={n_low_cal} HIGH={n_high_cal} total={len(rad_cal)} thr={thr:.2f}")
    if len(rad_cal) > 0:
        log.info(
            f"Calib true kWh: min={float(np.min(y_cal_true)):.2f} mean={float(np.mean(y_cal_true)):.2f} max={float(np.max(y_cal_true)):.2f}"
        )
        log.info(
            f"Calib rad:      min={float(np.min(rad_cal)):.2f} mean={float(np.mean(rad_cal)):.2f} max={float(np.max(rad_cal)):.2f}"
        )

    # Global fallback
    a_glob, b_glob = fit_linear_calibration(y_cal_pred_raw, y_cal_true, min_points=CALIB_MIN_POINTS)
    a_glob, b_glob = clip_calibration(a_glob, b_glob, "GLOBAL")
    log.info(f"Calib params (global): a={a_glob:.4f} b={b_glob:.4f} (min_points={CALIB_MIN_POINTS})")

    if n_low_cal >= CALIB_MIN_POINTS:
        a_low, b_low = fit_linear_calibration(y_cal_pred_raw[idx_low], y_cal_true[idx_low], min_points=CALIB_MIN_POINTS)
        a_low, b_low = clip_calibration(a_low, b_low, "LOW")
    else:
        a_low, b_low = a_glob, b_glob
        log.warning(f"För få LOW-punkter för segmentkalibrering (LOW={n_low_cal} < {CALIB_MIN_POINTS}) -> använder GLOBAL.")

    if n_high_cal >= CALIB_MIN_POINTS:
        a_high, b_high = fit_linear_calibration(y_cal_pred_raw[idx_high], y_cal_true[idx_high], min_points=CALIB_MIN_POINTS)
        a_high, b_high = clip_calibration(a_high, b_high, "HIGH")
    else:
        a_high, b_high = a_glob, b_glob
        log.warning(f"För få HIGH-punkter för segmentkalibrering (HIGH={n_high_cal} < {CALIB_MIN_POINTS}) -> använder GLOBAL.")

    log.info(f"Calib params (final): LOW a={a_low:.4f} b={b_low:.4f} | HIGH a={a_high:.4f} b={b_high:.4f}")

    # Apply calibration on calib for reporting
    y_cal_pred_cal = y_cal_pred_raw.copy()
    y_cal_pred_cal[idx_low] = apply_calibration(y_cal_pred_raw[idx_low], a_low, b_low)
    y_cal_pred_cal[idx_high] = apply_calibration(y_cal_pred_raw[idx_high], a_high, b_high)

    calib_mae_raw = mae(y_cal_true, y_cal_pred_raw)
    calib_mae_cal = mae(y_cal_true, y_cal_pred_cal)

    # -----------------------
    # Utvärdera på hela holdout (val_df)
    # -----------------------
    X_hold = val_df[feature_cols]
    y_hold_true = val_df["kwh"].values.astype(float)
    rad_hold = val_df[rad_col].values.astype(float)

    pred_low_hold = predict_kwh_from_ensemble(boosters_low, X_hold, feature_cols)
    pred_high_hold = predict_kwh_from_ensemble(boosters_high, X_hold, feature_cols)
    w_hold = gate_weight(rad_hold, thr, gate_scale)
    y_hold_pred_raw = blend_preds(pred_low_hold, pred_high_hold, w_hold)

    idx_low_h = rad_hold <= thr
    idx_high_h = ~idx_low_h

    y_hold_pred_cal = y_hold_pred_raw.copy()
    y_hold_pred_cal[idx_low_h] = apply_calibration(y_hold_pred_raw[idx_low_h], a_low, b_low)
    y_hold_pred_cal[idx_high_h] = apply_calibration(y_hold_pred_raw[idx_high_h], a_high, b_high)

    holdout_mae_raw = mae(y_hold_true, y_hold_pred_raw)
    holdout_mae_cal = mae(y_hold_true, y_hold_pred_cal)

    # Välj bästa och stäng av kalibrering om den försämrar
    holdout_mae_best = float(min(holdout_mae_raw, holdout_mae_cal))
    calibration_enabled = bool(holdout_mae_cal <= holdout_mae_raw)
    if not calibration_enabled:
        log.warning("Kalibrering försämrade holdout -> stänger av kalibrering för sparad modell (a=1,b=0).")
        a_low, b_low = 1.0, 0.0
        a_high, b_high = 1.0, 0.0

    # -----------------------
    # FINAL retrain ensemble på ALL data, per seed (best_iter+1)
    # -----------------------
    all_low = data[data[rad_col] <= thr]
    all_high = data[data[rad_col] > thr]

    final_boosters_low, final_boosters_high = [], []
    for i, (bi_low, bi_high) in enumerate(zip(best_iters_low, best_iters_high), 1):
        rounds_low = max(1, int(bi_low) + 1)
        rounds_high = max(1, int(bi_high) + 1)
        log.info(f"Final retrain ens[{i}/{len(best_iters_low)}]: LOW rounds={rounds_low} | HIGH rounds={rounds_high}")

        seed = int(params.get("seed", 42) + (i - 1) * 101)
        p = dict(params)
        p["seed"] = seed

        fl = retrain_final_booster_log1p(
            all_low[feature_cols],
            all_low["kwh"].values,
            feature_cols,
            p,
            device=device,
            num_round=rounds_low,
        )
        fh = retrain_final_booster_log1p(
            all_high[feature_cols],
            all_high["kwh"].values,
            feature_cols,
            p,
            device=device,
            num_round=rounds_high,
        )
        final_boosters_low.append(fl)
        final_boosters_high.append(fh)

    # -----------------------
    # Plot scan
    # -----------------------
    rows = coarse_rows + fine_rows
    plot_mae_vs_threshold(
        rows=rows,
        out_png=os.path.join("plots", "mae_vs_threshold.png"),
        out_csv=os.path.join("plots", "mae_vs_threshold.csv"),
    )

    # -----------------------
    # Spara modell – skriv inte över om befintlig är bättre
    # -----------------------
    model_payload = {
        # Backward-compat (för äldre inference-kod):
        "booster_low": final_boosters_low[0] if final_boosters_low else None,
        "booster_high": final_boosters_high[0] if final_boosters_high else None,

        # Nya nycklar (använd dessa för ensemble):
        "boosters_low": final_boosters_low,
        "boosters_high": final_boosters_high,
        "ensemble": int(len(final_boosters_low)),

        "feature_cols": feature_cols,
        "target_transform": "log1p_split",
        "radiation_threshold": float(thr),
        "gate": {"type": "sigmoid", "scale": float(gate_scale)},

        "best_iterations": {"low": best_iters_low, "high": best_iters_high},
        "calibration_low": {"a": float(a_low), "b": float(b_low)},
        "calibration_high": {"a": float(a_high), "b": float(b_high)},
        "calibration_enabled": bool(calibration_enabled),
        "calibration_clip": {"a": list(CALIB_A_CLIP), "b": list(CALIB_B_CLIP)},
        "calibration_min_points": int(CALIB_MIN_POINTS),

        "kwh_clip": float(args.kwh_clip),
        "snow": {
            "snow_kwh": float(args.snow_kwh),
            "snow_clear": float(args.snow_clear),
            "snow_window": int(args.snow_window),
        },

        "weather_cols": list(CFG.weather_daily_vars),
        "train_device": device,
        "early_stopping_metric": "mae_kwh",
    }

    out_path = os.path.join(CFG.models_dir, "TOTAL.joblib")
    report_path = os.path.join(CFG.models_dir, "model_report_total.json")

    prev_best = None
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                prev_report = json.load(f)
            if "holdout_MAE_best_kwh" in prev_report:
                prev_best = float(prev_report["holdout_MAE_best_kwh"])
            else:
                prev_raw = float(prev_report.get("holdout_MAE_raw_kwh", float("inf")))
                prev_cal = float(prev_report.get("holdout_MAE_cal_kwh", float("inf")))
                prev_best = float(min(prev_raw, prev_cal))
        except Exception as e:
            log.warning(f"Kunde inte läsa tidigare report ({report_path}): {e}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate_path = os.path.join(CFG.models_dir, f"TOTAL_candidate_{timestamp}.joblib")
    candidate_report_path = os.path.join(CFG.models_dir, f"model_report_total_candidate_{timestamp}.json")

    base_report = {
        "target": "TOTAL",
        "rows_used": int(len(data)),
        "holdout_days": int(holdout),
        "calib_days": int(calib_days),
        "train_device": device,
        "radiation_threshold": float(thr),
        "gate_scale": float(gate_scale),
        "ensemble": int(args.ensemble),
        "kwh_clip": float(args.kwh_clip),
        "snow": {"snow_kwh": float(args.snow_kwh), "snow_clear": float(args.snow_clear), "snow_window": int(args.snow_window)},
        "calib_MAE_raw_kwh": float(calib_mae_raw),
        "calib_MAE_cal_kwh": float(calib_mae_cal),
        "holdout_MAE_raw_kwh": float(holdout_mae_raw),
        "holdout_MAE_cal_kwh": float(holdout_mae_cal),
        "holdout_MAE_best_kwh": float(holdout_mae_best),
        "calibration_enabled": bool(calibration_enabled),
        "threshold_scan_rows": rows,
        "feature_count": int(len(feature_cols)),
    }

    if prev_best is not None and prev_best <= holdout_mae_best:
        log.warning(
            f"❌ Ny modell sämre eller lika bra (best_MAE={holdout_mae_best:.3f}) än befintlig (best_MAE={prev_best:.3f}). Skriver INTE över."
        )
        base_report["model_file"] = candidate_path
        joblib.dump(model_payload, candidate_path)
        with open(candidate_report_path, "w", encoding="utf-8") as f:
            json.dump(base_report, f, ensure_ascii=False, indent=2)
        saved_model_path = candidate_path
        saved_report_path = candidate_report_path
    else:
        if prev_best is None:
            log.info("✅ Ingen tidigare modell hittad – sparar denna som aktuell.")
        else:
            log.info(f"✅ Ny modell bättre (best_MAE={holdout_mae_best:.3f} < {prev_best:.3f}) – skriver över aktuell.")
        base_report["model_file"] = out_path
        joblib.dump(model_payload, out_path)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(base_report, f, ensure_ascii=False, indent=2)
        saved_model_path = out_path
        saved_report_path = report_path

    log.info(f"Sparade modell: {saved_model_path}")
    log.info(f"Sparade report: {saved_report_path}")

    log.info(
        f"Vald threshold={thr:.2f} | holdout_MAE_raw={holdout_mae_raw:.3f} | holdout_MAE_cal={holdout_mae_cal:.3f} | device={device}"
    )
    if calib_days > 0:
        log.info(f"Kalib (fit-set) MAE_raw={calib_mae_raw:.3f} | MAE_cal={calib_mae_cal:.3f} (OBS fit på calib)")
    log.info(f"Klart. Total tid: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
