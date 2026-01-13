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
# Helpers
# -----------------------
def day_length_hours(date_ts: pd.Timestamp, latitude_deg: float) -> float:
    n = int(date_ts.dayofyear)
    lat = math.radians(latitude_deg)
    decl = math.radians(23.44) * math.sin(math.radians((360 / 365.0) * (n - 81)))
    cos_omega = -math.tan(lat) * math.tan(decl)
    cos_omega = max(-1.0, min(1.0, cos_omega))
    omega = math.acos(cos_omega)
    return (2 * omega * 24) / (2 * math.pi)


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values("date").copy()

    df["dow"] = df["date"].dt.dayofweek
    df["doy"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df["daylight_h"] = df["date"].apply(lambda d: day_length_hours(pd.Timestamp(d), CFG.latitude))

    # Historik
    lags = (1, 2, 7)
    for lag in lags:
        df[f"kwh_lag{lag}"] = df["kwh"].shift(lag)

    windows = (7, 14, 28)
    for w in windows:
        df[f"kwh_ma{w}"] = df["kwh"].rolling(window=w, min_periods=max(3, w // 2)).mean().shift(1)

    # Extra historik-signal (brukar hjälpa)
    for span in (7, 14, 28):
        df[f"kwh_ewm{span}"] = df["kwh"].shift(1).ewm(span=span, adjust=False).mean()
        df[f"kwh_std{span}"] = df["kwh"].shift(1).rolling(window=span, min_periods=max(3, span // 2)).std()

    # Interaktion (brukar hjälpa lite)
    if "shortwave_radiation_sum" in df.columns:
        df["rad_x_daylight"] = df["shortwave_radiation_sum"] * df["daylight_h"]

    base_cols = ["dow", "doy", "sin_doy", "cos_doy", "daylight_h"]
    hist_cols = (
        [f"kwh_lag{lag}" for lag in lags]
        + [f"kwh_ma{w}" for w in windows]
        + [f"kwh_ewm{span}" for span in (7, 14, 28)]
        + [f"kwh_std{span}" for span in (7, 14, 28)]
    )
    weather_cols = list(CFG.weather_daily_vars)

    feature_cols = base_cols + hist_cols + weather_cols
    if "rad_x_daylight" in df.columns:
        feature_cols.append("rad_x_daylight")

    df = df.dropna(subset=feature_cols + ["kwh"]).copy()
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols + ["kwh"]).copy()

    return df, feature_cols


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def fit_linear_calibration(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if len(y_pred) < 5:
        return 1.0, 0.0

    var = float(np.var(y_pred))
    if var < 1e-9:
        return 1.0, 0.0

    cov = float(np.mean((y_pred - y_pred.mean()) * (y_true - y_true.mean())))
    a = cov / var
    b = float(y_true.mean() - a * y_pred.mean())
    return a, b


def apply_calibration(y_pred: np.ndarray, a: float, b: float) -> np.ndarray:
    y = a * np.asarray(y_pred, dtype=float) + b
    return np.clip(y, 0.0, None)


# -----------------------
# Custom eval: early stopping på MAE i kWh (inte log-domän)
# -----------------------
def mae_kwh_from_logpred(predt: np.ndarray, dmat: xgb.DMatrix):
    """
    predt: pred i log1p(kWh)
    label: true i log1p(kWh)
    Returnerar MAE i kWh, så early stopping optimerar rätt sak.
    """
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
    """
    Träna booster på log1p(kWh), med early stopping på MAE i kWh.
    """
    y_tr_log = np.log1p(np.asarray(y_tr_kwh, dtype=float))
    y_val_log = np.log1p(np.asarray(y_val_kwh, dtype=float))

    dtrain = xgb.DMatrix(X_tr, label=y_tr_log, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val_log, feature_names=feature_cols)

    params_dev = dict(params)
    params_dev["device"] = device
    params_dev["disable_default_eval_metric"] = 1  # undvik default metrics i log-domän

    # XGBoost >= 2.0: custom_metric, annars fallback feval
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
    """
    Retraina final-booster på ALL data med fast num_boost_round (best_iter+1).
    Ingen early stopping (för att nyttja all data).
    """
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


def parse_args():
    p = argparse.ArgumentParser(description="Träna PV-modell (LOW/HIGH split) med threshold-scan och logging.")
    p.add_argument(
        "--verbose-eval",
        type=int,
        default=0,
        help="XGBoost verbose_eval. 0 = tyst. T.ex. 200 för progress.",
    )
    p.add_argument(
        "--holdout",
        type=int,
        default=60,
        help="Antal dagar i holdout (val). Default 60.",
    )
    p.add_argument(
        "--calib-days",
        type=int,
        default=0,
        help="Om >0: sista N dagar av holdout används för kalibrering (threshold väljs på resterande holdout).",
    )
    return p.parse_args()


def main():
    ensure_dir(CFG.models_dir)
    ensure_dir("plots")

    args = parse_args()
    log_system_info()

    device = "cpu"
    cpu_threads = int(os.environ.get("XGBOOST_NUM_THREADS", os.cpu_count() or 1))
    log.info(f"Vald device: {device} | CPU-trådar: {cpu_threads}")
    t0 = time.time()
    log.info("Startar träning: SPLIT (LOW/HIGH) + threshold-scan + MAE-graf + kalibrering + FINAL retrain")

    prod = pd.read_csv("data/production_total.csv", parse_dates=["date"]).sort_values("date")
    weather = pd.read_csv(CFG.weather_historical_path, parse_dates=["date"]).sort_values("date")
    data = prod.merge(weather, on="date", how="inner")

    if data.empty:
        raise SystemExit("Ingen överlappning mellan produktion och väder.")

    data, feature_cols = make_features(data)
    log.info(f"Rader efter features: {len(data)} | Antal features: {len(feature_cols)}")

    # Holdout (sista N dagar)
    holdout = min(args.holdout, max(21, int(0.2 * len(data))))
    train_df = data.iloc[:-holdout].copy()
    val_df = data.iloc[-holdout:].copy()
    log.info(f"Holdout={holdout} | Train={len(train_df)} | Holdout(total)={len(val_df)}")

    # (Valfritt) dela holdout i val_select + calib
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

    # Stabil params (bra baseline)
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
    verbose_eval = args.verbose_eval

    # Kandidater: coarse + fine
    coarse_pcts = list(range(20, 81, 5))
    train_rad = train_df[rad_col].values.astype(float)
    coarse_thrs = [float(np.percentile(train_rad, p)) for p in coarse_pcts]
    log.info(f"Coarse scan percentiler: {coarse_pcts}")

    def eval_thr(thr: float):
        tr_low = train_df[train_df[rad_col] <= thr]
        tr_high = train_df[train_df[rad_col] > thr]
        va_low = val_select_df[val_select_df[rad_col] <= thr]
        va_high = val_select_df[val_select_df[rad_col] > thr]

        # Eval-set per del (om för liten, använd hela val_select)
        X_val_low = (va_low[feature_cols] if len(va_low) >= 10 else val_select_df[feature_cols])
        y_val_low = (va_low["kwh"].values if len(va_low) >= 10 else val_select_df["kwh"].values)

        X_val_high = (va_high[feature_cols] if len(va_high) >= 10 else val_select_df[feature_cols])
        y_val_high = (va_high["kwh"].values if len(va_high) >= 10 else val_select_df["kwh"].values)

        booster_low = train_booster_log1p(
            tr_low[feature_cols],
            tr_low["kwh"].values,
            X_val_low,
            y_val_low,
            feature_cols,
            params,
            num_round,
            es_rounds,
            device=device,
            verbose_eval=verbose_eval,
        )

        booster_high = train_booster_log1p(
            tr_high[feature_cols],
            tr_high["kwh"].values,
            X_val_high,
            y_val_high,
            feature_cols,
            params,
            num_round,
            es_rounds,
            device=device,
            verbose_eval=verbose_eval,
        )

        # logga att den tränar och var den stannade
        try:
            log.info(f"  LOW  best_iter={booster_low.best_iteration} best_score={booster_low.best_score}")
            log.info(f"  HIGH best_iter={booster_high.best_iteration} best_score={booster_high.best_score}")
        except Exception:
            pass

        # Score på val_select (för threshold-val)
        X_va = val_select_df[feature_cols]
        rad_va = val_select_df[rad_col].values.astype(float)
        y_true = val_select_df["kwh"].values.astype(float)

        pred_low = predict_kwh_from_booster(booster_low, X_va, feature_cols)
        pred_high = predict_kwh_from_booster(booster_high, X_va, feature_cols)

        y_pred = np.where(rad_va <= thr, pred_low, pred_high)
        score = mae(y_true, y_pred)

        return score, booster_low, booster_high

    best = None
    coarse_rows = []
    for i, (p, thr) in enumerate(zip(coarse_pcts, coarse_thrs), 1):
        log.info(f"[Coarse {i}/{len(coarse_thrs)}] pct={p} thr={thr:.2f}")
        t_thr = time.time()
        score, bl, bh = eval_thr(thr)
        coarse_rows.append(
            {"phase": "coarse", "pct": p, "thr": thr, "mae": score, "seconds": round(time.time() - t_thr, 2)}
        )
        log.info(f"Coarse thr={thr:.2f} (pct={p}) | MAE={score:.3f} | {time.time() - t_thr:.1f}s")
        if best is None or score < best["mae"]:
            best = {"thr": float(thr), "mae": float(score), "booster_low": bl, "booster_high": bh}
            log.info(f"✅ Ny bästa (coarse): thr={thr:.2f} | MAE={score:.3f}")

    # Fine scan runt bästa coarse
    best_thr = best["thr"]
    best_pct = float(np.mean(train_rad <= best_thr) * 100.0)
    fine_pcts = list(range(max(5, int(best_pct) - 8), min(95, int(best_pct) + 8) + 1, 1))
    fine_thrs = [float(np.percentile(train_rad, p)) for p in fine_pcts]
    log.info(f"Fine scan runt pct≈{best_pct:.1f}: percentiler {fine_pcts[0]}..{fine_pcts[-1]}")

    fine_rows = []
    for i, (p, thr) in enumerate(zip(fine_pcts, fine_thrs), 1):
        log.info(f"[Fine {i}/{len(fine_thrs)}] pct={p} thr={thr:.2f}")
        t_thr = time.time()
        score, bl, bh = eval_thr(thr)
        fine_rows.append(
            {"phase": "fine", "pct": p, "thr": thr, "mae": score, "seconds": round(time.time() - t_thr, 2)}
        )
        log.info(f"Fine thr={thr:.2f} (pct={p}) | MAE={score:.3f} | {time.time() - t_thr:.1f}s")
        if score < best["mae"]:
            best = {"thr": float(thr), "mae": float(score), "booster_low": bl, "booster_high": bh}
            log.info(f"✅ Ny bästa (fine): thr={thr:.2f} | MAE={score:.3f}")

    thr = best["thr"]

    # -----------------------
    # Kalibrering (fit på calib_df)
    # -----------------------
    X_cal = calib_df[feature_cols]
    y_cal_true = calib_df["kwh"].values.astype(float)
    rad_cal = calib_df[rad_col].values.astype(float)

    pred_low_cal = predict_kwh_from_booster(best["booster_low"], X_cal, feature_cols)
    pred_high_cal = predict_kwh_from_booster(best["booster_high"], X_cal, feature_cols)
    y_cal_pred_raw = np.where(rad_cal <= thr, pred_low_cal, pred_high_cal)

    idx_low = rad_cal <= thr
    idx_high = ~idx_low

    a_low, b_low = fit_linear_calibration(y_cal_pred_raw[idx_low], y_cal_true[idx_low]) if np.any(idx_low) else (1.0, 0.0)
    a_high, b_high = fit_linear_calibration(y_cal_pred_raw[idx_high], y_cal_true[idx_high]) if np.any(idx_high) else (1.0, 0.0)

    y_cal_pred_cal = y_cal_pred_raw.copy()
    y_cal_pred_cal[idx_low] = apply_calibration(y_cal_pred_raw[idx_low], a_low, b_low)
    y_cal_pred_cal[idx_high] = apply_calibration(y_cal_pred_raw[idx_high], a_high, b_high)

    calib_mae_raw = mae(y_cal_true, y_cal_pred_raw)
    calib_mae_cal = mae(y_cal_true, y_cal_pred_cal)

    # -----------------------
    # Utvärdera på hela holdout (val_df) med kalibrering som fitted ovan
    # -----------------------
    X_hold = val_df[feature_cols]
    y_hold_true = val_df["kwh"].values.astype(float)
    rad_hold = val_df[rad_col].values.astype(float)

    pred_low_hold = predict_kwh_from_booster(best["booster_low"], X_hold, feature_cols)
    pred_high_hold = predict_kwh_from_booster(best["booster_high"], X_hold, feature_cols)
    y_hold_pred_raw = np.where(rad_hold <= thr, pred_low_hold, pred_high_hold)

    idx_low_h = rad_hold <= thr
    idx_high_h = ~idx_low_h

    y_hold_pred_cal = y_hold_pred_raw.copy()
    y_hold_pred_cal[idx_low_h] = apply_calibration(y_hold_pred_raw[idx_low_h], a_low, b_low)
    y_hold_pred_cal[idx_high_h] = apply_calibration(y_hold_pred_raw[idx_high_h], a_high, b_high)

    holdout_mae_raw = mae(y_hold_true, y_hold_pred_raw)
    holdout_mae_cal = mae(y_hold_true, y_hold_pred_cal)

    # -----------------------
    # FINAL retrain på ALL data med best_iteration+1
    # -----------------------
    best_iter_low = int(getattr(best["booster_low"], "best_iteration", 0))
    best_iter_high = int(getattr(best["booster_high"], "best_iteration", 0))
    final_rounds_low = max(1, best_iter_low + 1)
    final_rounds_high = max(1, best_iter_high + 1)

    all_low = data[data[rad_col] <= thr]
    all_high = data[data[rad_col] > thr]

    log.info(f"Final retrain: LOW rounds={final_rounds_low} | HIGH rounds={final_rounds_high}")
    final_low = retrain_final_booster_log1p(
        all_low[feature_cols],
        all_low["kwh"].values,
        feature_cols,
        params,
        device=device,
        num_round=final_rounds_low,
    )
    final_high = retrain_final_booster_log1p(
        all_high[feature_cols],
        all_high["kwh"].values,
        feature_cols,
        params,
        device=device,
        num_round=final_rounds_high,
    )

    # Spara graf + csv
    rows = coarse_rows + fine_rows
    plot_mae_vs_threshold(
        rows=rows,
        out_png=os.path.join("plots", "mae_vs_threshold.png"),
        out_csv=os.path.join("plots", "mae_vs_threshold.csv"),
    )

    # Spara modell (sparar FINAL boosters)
    out_path = os.path.join(CFG.models_dir, "TOTAL.joblib")
    joblib.dump(
        {
            "booster_low": final_low,
            "booster_high": final_high,
            "feature_cols": feature_cols,
            "target_transform": "log1p_split",
            "radiation_threshold": thr,
            "best_iterations": {"low": best_iter_low, "high": best_iter_high},
            "final_rounds": {"low": final_rounds_low, "high": final_rounds_high},
            "calibration_low": {"a": a_low, "b": b_low},
            "calibration_high": {"a": a_high, "b": b_high},
            "weather_cols": list(CFG.weather_daily_vars),
            "train_device": device,
            "early_stopping_metric": "mae_kwh",
        },
        out_path,
    )

    report = {
        "target": "TOTAL",
        "rows_used": int(len(data)),
        "holdout_days": int(holdout),
        "calib_days": int(calib_days),
        "train_device": device,
        "radiation_threshold": float(thr),
        "best_iterations": {"low": best_iter_low, "high": best_iter_high},
        "final_rounds": {"low": final_rounds_low, "high": final_rounds_high},
        "calib_MAE_raw_kwh": float(calib_mae_raw),
        "calib_MAE_cal_kwh": float(calib_mae_cal),
        "holdout_MAE_raw_kwh": float(holdout_mae_raw),
        "holdout_MAE_cal_kwh": float(holdout_mae_cal),
        "model_file": out_path,
        "threshold_scan_rows": rows,
        "feature_count": int(len(feature_cols)),
    }
    with open(os.path.join(CFG.models_dir, "model_report_total.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info(f"Sparade modell: {out_path}")
    log.info(
        f"Vald threshold={thr:.2f} | holdout_MAE_raw={holdout_mae_raw:.3f} | holdout_MAE_cal={holdout_mae_cal:.3f} | device={device}"
    )
    if calib_days > 0:
        log.info(f"Kalib (fit-set) MAE_raw={calib_mae_raw:.3f} | MAE_cal={calib_mae_cal:.3f} (OBS fit på calib)")
    log.info(f"Klart. Total tid: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
