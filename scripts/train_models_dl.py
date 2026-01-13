# scripts/train_models_dl.py
# Deep learning-version + multitrådning.
# - DataLoader: num_workers, persistent_workers, prefetch_factor, pin_memory
# - Torch CPU threads: torch.set_num_threads / torch.set_num_interop_threads
# OBS (Windows): num_workers>0 kräver att du kör via "python -m ..." eller direktfil
# och att allt ligger under if __name__ == "__main__": (vilket det gör här).

from __future__ import annotations

import os
import json
import math
import time
import logging
import argparse
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import CFG
from scripts._common import ensure_dir

# -----------------------
# Torch (kräver: pip install torch)
# -----------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "PyTorch saknas. Installera först: pip install torch\n"
        f"Originalfel: {e}"
    )


# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_dl")


# -----------------------
# Safeguards / defaults
# -----------------------
CALIB_MIN_POINTS = int(os.environ.get("CALIB_MIN_POINTS", "20"))
CALIB_A_CLIP = (0.7, 1.3)
CALIB_B_CLIP = (-3.0, 3.0)

DEFAULT_KWH_CLIP = float(os.environ.get("KWH_CLIP", "35"))
RAD_COL = "shortwave_radiation_sum"


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    """FAO-56 Ra (MJ/m^2/day)."""
    Gsc = 0.0820
    n = int(date_ts.dayofyear)
    phi = math.radians(latitude_deg)

    dr = 1.0 + 0.033 * math.cos((2.0 * math.pi / 365.0) * n)
    delta = 0.409 * math.sin((2.0 * math.pi / 365.0) * n - 1.39)

    cos_ws = -math.tan(phi) * math.tan(delta)
    cos_ws = max(-1.0, min(1.0, cos_ws))
    ws = math.acos(cos_ws)

    Ra = (
        (24.0 * 60.0 / math.pi)
        * Gsc
        * dr
        * (ws * math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.sin(ws))
    )
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

    if kwh_clip and float(kwh_clip) > 0:
        df["kwh"] = df["kwh"].clip(lower=0.0, upper=float(kwh_clip))

    df["dow"] = df["date"].dt.dayofweek
    df["doy"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df["daylight_h"] = df["date"].apply(lambda d: day_length_hours(pd.Timestamp(d), CFG.latitude))

    df["ra_MJ"] = df["date"].apply(lambda d: extraterrestrial_radiation_MJ_m2_day(pd.Timestamp(d), CFG.latitude))

    rad_col = RAD_COL
    has_rad = rad_col in df.columns
    if has_rad:
        eps = 1e-6
        df["clear_idx"] = (df[rad_col] / (df["ra_MJ"] + eps)).clip(0.0, 1.5)

        df["rad_lag1"] = df[rad_col].shift(1)
        df["rad_ma3"] = df[rad_col].shift(1).rolling(3, min_periods=2).mean()
        df["rad_ma7"] = df[rad_col].shift(1).rolling(7, min_periods=4).mean()

        df["clear_idx_lag1"] = df["clear_idx"].shift(1)
        df["clear_idx_ma3"] = df["clear_idx"].shift(1).rolling(3, min_periods=2).mean()

        df["rad_x_daylight"] = df[rad_col] * df["daylight_h"]
        df["rad_x_clear"] = df[rad_col] * df["clear_idx"]

    lags = (1, 2, 7)
    for lag in lags:
        df[f"kwh_lag{lag}"] = df["kwh"].shift(lag)

    windows = (7, 14, 28)
    for w in windows:
        df[f"kwh_ma{w}"] = df["kwh"].rolling(window=w, min_periods=max(3, w // 2)).mean().shift(1)

    for span in (7, 14, 28):
        df[f"kwh_ewm{span}"] = df["kwh"].shift(1).ewm(span=span, adjust=False).mean()
        df[f"kwh_std{span}"] = df["kwh"].shift(1).rolling(window=span, min_periods=max(3, span // 2)).std()

    if has_rad and "clear_idx_lag1" in df.columns:
        df["snow_flag"] = (
            (df["kwh_lag1"].fillna(0.0) <= float(snow_kwh))
            & (df["clear_idx_lag1"].fillna(0.0) >= float(snow_clear))
        ).astype(int)
    elif has_rad and "rad_lag1" in df.columns:
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

    df = df.dropna(subset=feature_cols + ["kwh"]).copy()
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols + ["kwh"]).copy()

    return df, feature_cols


# -----------------------
# Scaler
# -----------------------
@dataclass
class StandardScalerNP:
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-6) -> "StandardScalerNP":
        X = np.asarray(X, dtype=np.float32)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean_=mean.astype(np.float32), std_=std.astype(np.float32))

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / self.std_


# -----------------------
# Dataset
# -----------------------
class TabularPvDataset(Dataset):
    def __init__(self, X: np.ndarray, rad: np.ndarray, y_kwh: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.rad = np.asarray(rad, dtype=np.float32)
        self.y = np.asarray(y_kwh, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.rad[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


# -----------------------
# Model
# -----------------------
class MLPExpert(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.10):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedSigmoidGate(nn.Module):
    def __init__(self, init_thr: float, init_scale: float = 0.25):
        super().__init__()
        self.thr = nn.Parameter(torch.tensor(float(init_thr), dtype=torch.float32))
        self.log_scale = nn.Parameter(torch.tensor(math.log(max(1e-3, float(init_scale))), dtype=torch.float32))
        self.softplus = nn.Softplus()

    def forward(self, rad: torch.Tensor) -> torch.Tensor:
        scale = self.softplus(self.log_scale) + 1e-3
        z = (rad - self.thr) / scale
        z = torch.clamp(z, -30.0, 30.0)
        return torch.sigmoid(z)


class PvMoE(nn.Module):
    def __init__(self, in_dim: int, init_thr: float, init_scale: float, hidden=(128, 64), dropout=0.10):
        super().__init__()
        self.expert_low = MLPExpert(in_dim, hidden=tuple(hidden), dropout=float(dropout))
        self.expert_high = MLPExpert(in_dim, hidden=tuple(hidden), dropout=float(dropout))
        self.gate = LearnedSigmoidGate(init_thr=float(init_thr), init_scale=float(init_scale))

    def forward(self, x: torch.Tensor, rad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p_low = self.expert_low(x).squeeze(-1)
        p_high = self.expert_high(x).squeeze(-1)
        w = self.gate(rad)
        p = (1.0 - w) * p_low + w * p_high
        return p, w


# -----------------------
# Train / eval
# -----------------------
@torch.no_grad()
def eval_mae_kwh(model: PvMoE, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    ys, preds = [], []
    for xb, radb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        radb = radb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred_log, _ = model(xb, radb)
        pred_kwh = torch.expm1(pred_log).clamp(min=0.0)
        ys.append(yb.detach().cpu().numpy())
        preds.append(pred_kwh.detach().cpu().numpy())
    y = np.concatenate(ys) if ys else np.array([], dtype=float)
    p = np.concatenate(preds) if preds else np.array([], dtype=float)
    return mae(y, p)


def train_one(
    *,
    seed: int,
    model: PvMoE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    grad_clip: float,
    verbose_batches: int = 0,
) -> tuple[PvMoE, dict]:
    set_seed(seed)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0

    history = {"epoch": [], "train_mae": [], "val_mae": [], "thr": [], "scale": []}

    for epoch in range(1, int(epochs) + 1):
        t0 = time.time()
        model.train()

        running = 0.0
        n_seen = 0
        for b, (xb, radb, yb) in enumerate(train_loader, 1):
            xb = xb.to(device, non_blocking=True)
            radb = radb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred_log, _ = model(xb, radb)
            pred_kwh = torch.expm1(pred_log).clamp(min=0.0)
            loss = torch.mean(torch.abs(pred_kwh - yb))
            loss.backward()

            if grad_clip and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

            opt.step()

            running += float(loss.detach().item()) * int(len(yb))
            n_seen += int(len(yb))

            if verbose_batches and (b % int(verbose_batches) == 0):
                log.info(f"  epoch {epoch} batch {b}/{len(train_loader)} | loss={float(loss.detach().item()):.4f}")

        train_mae = float(running / max(1, n_seen))
        val_mae = float(eval_mae_kwh(model, val_loader, device))

        thr = float(model.gate.thr.detach().cpu().item())
        scale = float((model.gate.softplus(model.gate.log_scale) + 1e-3).detach().cpu().item())

        history["epoch"].append(epoch)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["thr"].append(thr)
        history["scale"].append(scale)

        dt = time.time() - t0
        log.info(
            f"seed={seed} | epoch {epoch:03d}/{epochs} | train_MAE={train_mae:.3f} kWh | val_MAE={val_mae:.3f} kWh | "
            f"gate thr={thr:.2f} scale={scale:.3f} | {dt:.1f}s"
        )

        improved = val_mae < (best_val - 1e-4)
        if improved:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= int(patience):
            log.info(
                f"Early stopping: ingen förbättring på {patience} epoker. "
                f"Bästa epoch={best_epoch} val_MAE={best_val:.3f}."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_mae": float(best_val), "best_epoch": int(best_epoch), "history": history}


@torch.no_grad()
def predict_ensemble_kwh(models: list[PvMoE], X: np.ndarray, rad: np.ndarray, device: torch.device, batch_size: int, num_workers: int) -> np.ndarray:
    if not models:
        return np.zeros(len(X), dtype=float)

    X = np.asarray(X, dtype=np.float32)
    rad = np.asarray(rad, dtype=np.float32)

    ds = TabularPvDataset(X, rad, np.zeros(len(X), dtype=np.float32))

    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(num_workers) > 0),
        prefetch_factor=(2 if int(num_workers) > 0 else None),
    )

    preds_all = []
    for m in models:
        m.eval()
        m.to(device)

        preds = []
        for xb, radb, _ in dl:
            xb = xb.to(device, non_blocking=True)
            radb = radb.to(device, non_blocking=True)
            pred_log, _ = m(xb, radb)
            pred_kwh = torch.expm1(pred_log).clamp(min=0.0)
            preds.append(pred_kwh.detach().cpu().numpy())
        preds_all.append(np.concatenate(preds))

    return np.mean(np.stack(preds_all, axis=0), axis=0)


# -----------------------
# Plot
# -----------------------
def plot_learning_curve(histories: list[dict], out_png: str):
    plt.figure(figsize=(10, 6))
    for i, h in enumerate(histories, 1):
        ep = h["epoch"]
        plt.plot(ep, h["val_mae"], marker="o", linewidth=1, label=f"ens{i} val")
    plt.title("DL learning curve (val MAE i kWh)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (kWh)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    log.info(f"Skrev graf: {out_png}")


# -----------------------
# Args
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Träna PV-modell med Deep Learning (PyTorch MoE) + multitrådning (DataLoader workers + torch threads)."
    )
    p.add_argument("--holdout", type=int, default=60)
    p.add_argument("--calib-days", type=int, default=35)
    p.add_argument("--ensemble", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--hidden", type=str, default="128,64")
    p.add_argument("--dropout", type=float, default=0.10)

    p.add_argument("--kwh-clip", type=float, default=DEFAULT_KWH_CLIP)
    p.add_argument("--snow-kwh", type=float, default=0.5)
    p.add_argument("--snow-clear", type=float, default=0.45)
    p.add_argument("--snow-window", type=int, default=7)

    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    p.add_argument("--verbose-batches", type=int, default=0)

    # --- Multitrådning / parallellism ---
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0=ingen multiprocessing). Prova 2-8 på CPU. Windows: börja med 2.",
    )
    p.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="torch.set_num_threads (0=auto). För CPU-träning: prova t.ex. antal fysiska kärnor.",
    )
    p.add_argument(
        "--torch-interop-threads",
        type=int,
        default=0,
        help="torch.set_num_interop_threads (0=auto). Ofta bra med 1-4.",
    )

    return p.parse_args()


# -----------------------
# Main
# -----------------------
def main():
    ensure_dir(CFG.models_dir)
    ensure_dir("plots")

    args = parse_args()

    # Torch threads (CPU)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if int(args.torch_interop_threads) > 0:
        torch.set_num_interop_threads(int(args.torch_interop_threads))

    log.info(
        "Torch threads: "
        f"num_threads={torch.get_num_threads()} | num_interop_threads={torch.get_num_interop_threads()}"
    )

    # Device
    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden = tuple(int(x.strip()) for x in str(args.hidden).split(",") if x.strip())
    if not hidden:
        hidden = (128, 64)

    t0 = time.time()
    log.info(f"Device: {device}")
    log.info(
        f"Startar DL-träning: MoE + ensemble(n={args.ensemble}) + DataLoader workers={args.num_workers} + snö-features"
    )

    # Ladda data
    prod = pd.read_csv("data/production_total.csv", parse_dates=["date"]).sort_values("date")
    weather = pd.read_csv(CFG.weather_historical_path, parse_dates=["date"]).sort_values("date")
    data = prod.merge(weather, on="date", how="inner")

    if data.empty:
        raise SystemExit("Ingen överlappning mellan produktion och väder.")

    if RAD_COL not in data.columns:
        raise SystemExit(f"Saknar {RAD_COL} i väderdata.")

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

    # Skala features (fit på train)
    X_tr_raw = train_df[feature_cols].values.astype(np.float32)
    scaler = StandardScalerNP.fit(X_tr_raw)

    X_tr = scaler.transform(X_tr_raw)
    y_tr = train_df["kwh"].values.astype(np.float32)
    rad_tr = train_df[RAD_COL].values.astype(np.float32)

    X_va = scaler.transform(val_select_df[feature_cols].values.astype(np.float32))
    y_va = val_select_df["kwh"].values.astype(np.float32)
    rad_va = val_select_df[RAD_COL].values.astype(np.float32)

    ds_tr = TabularPvDataset(X_tr, rad_tr, y_tr)
    ds_va = TabularPvDataset(X_va, rad_va, y_va)

    bs = int(args.batch_size)
    nw = int(args.num_workers)

    train_loader = DataLoader(
        ds_tr,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
        prefetch_factor=(2 if nw > 0 else None),
    )
    val_loader = DataLoader(
        ds_va,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
        prefetch_factor=(2 if nw > 0 else None),
    )

    # Gate init
    init_thr = float(np.median(rad_tr))
    init_scale = float(max(0.05, 0.25 * np.std(rad_tr)))

    models: list[PvMoE] = []
    reports: list[dict] = []
    histories: list[dict] = []

    for i in range(int(max(1, args.ensemble))):
        seed = int(args.seed + i * 101)
        log.info(f"\n=== Tränar ensemble {i+1}/{args.ensemble} | seed={seed} ===")

        set_seed(seed)
        model = PvMoE(
            in_dim=int(len(feature_cols)),
            init_thr=init_thr,
            init_scale=init_scale,
            hidden=hidden,
            dropout=float(args.dropout),
        )

        model, rep = train_one(
            seed=seed,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            patience=int(args.patience),
            grad_clip=float(args.grad_clip),
            verbose_batches=int(args.verbose_batches),
        )
        models.append(model)
        reports.append(rep)
        histories.append(rep["history"])

        thr_i = float(model.gate.thr.detach().cpu().item())
        scale_i = float((model.gate.softplus(model.gate.log_scale) + 1e-3).detach().cpu().item())
        log.info(
            f"ens {i+1}/{args.ensemble} klar | best_val_mae={rep['best_val_mae']:.3f} | thr={thr_i:.2f} scale={scale_i:.3f}"
        )

    # Kalibrering på calib_df
    X_cal = scaler.transform(calib_df[feature_cols].values.astype(np.float32))
    y_cal_true = calib_df["kwh"].values.astype(float)
    rad_cal = calib_df[RAD_COL].values.astype(float)

    y_cal_pred_raw = predict_ensemble_kwh(
        models, X_cal, rad_cal, device=device, batch_size=bs, num_workers=nw
    )

    thrs = [float(m.gate.thr.detach().cpu().item()) for m in models]
    thr_med = float(np.median(thrs))

    idx_low = rad_cal <= thr_med
    idx_high = ~idx_low
    n_low_cal = int(np.sum(idx_low))
    n_high_cal = int(np.sum(idx_high))

    log.info(f"Calib points: LOW={n_low_cal} HIGH={n_high_cal} total={len(rad_cal)} thr(median)={thr_med:.2f}")

    a_glob, b_glob = fit_linear_calibration(y_cal_pred_raw, y_cal_true, min_points=CALIB_MIN_POINTS)
    a_glob, b_glob = clip_calibration(a_glob, b_glob, "GLOBAL")

    if n_low_cal >= CALIB_MIN_POINTS:
        a_low, b_low = fit_linear_calibration(y_cal_pred_raw[idx_low], y_cal_true[idx_low], min_points=CALIB_MIN_POINTS)
        a_low, b_low = clip_calibration(a_low, b_low, "LOW")
    else:
        a_low, b_low = a_glob, b_glob
        log.warning(f"För få LOW-punkter (LOW={n_low_cal} < {CALIB_MIN_POINTS}) -> GLOBAL.")

    if n_high_cal >= CALIB_MIN_POINTS:
        a_high, b_high = fit_linear_calibration(y_cal_pred_raw[idx_high], y_cal_true[idx_high], min_points=CALIB_MIN_POINTS)
        a_high, b_high = clip_calibration(a_high, b_high, "HIGH")
    else:
        a_high, b_high = a_glob, b_glob
        log.warning(f"För få HIGH-punkter (HIGH={n_high_cal} < {CALIB_MIN_POINTS}) -> GLOBAL.")

    y_cal_pred_cal = y_cal_pred_raw.copy()
    y_cal_pred_cal[idx_low] = apply_calibration(y_cal_pred_raw[idx_low], a_low, b_low)
    y_cal_pred_cal[idx_high] = apply_calibration(y_cal_pred_raw[idx_high], a_high, b_high)

    calib_mae_raw = mae(y_cal_true, y_cal_pred_raw)
    calib_mae_cal = mae(y_cal_true, y_cal_pred_cal)

    # Holdout eval
    X_hold = scaler.transform(val_df[feature_cols].values.astype(np.float32))
    y_hold_true = val_df["kwh"].values.astype(float)
    rad_hold = val_df[RAD_COL].values.astype(float)

    y_hold_pred_raw = predict_ensemble_kwh(
        models, X_hold, rad_hold, device=device, batch_size=bs, num_workers=nw
    )

    idx_low_h = rad_hold <= thr_med
    idx_high_h = ~idx_low_h

    y_hold_pred_cal = y_hold_pred_raw.copy()
    y_hold_pred_cal[idx_low_h] = apply_calibration(y_hold_pred_raw[idx_low_h], a_low, b_low)
    y_hold_pred_cal[idx_high_h] = apply_calibration(y_hold_pred_raw[idx_high_h], a_high, b_high)

    holdout_mae_raw = mae(y_hold_true, y_hold_pred_raw)
    holdout_mae_cal = mae(y_hold_true, y_hold_pred_cal)

    holdout_mae_best = float(min(holdout_mae_raw, holdout_mae_cal))
    calibration_enabled = bool(holdout_mae_cal <= holdout_mae_raw)
    if not calibration_enabled:
        log.warning("Kalibrering försämrade holdout -> stänger av (a=1,b=0) i payload.")
        a_low, b_low = 1.0, 0.0
        a_high, b_high = 1.0, 0.0

    plot_learning_curve(histories, out_png=os.path.join("plots", "dl_learning_curve.png"))

    out_model = os.path.join(CFG.models_dir, "TOTAL_DL.pt")
    out_report = os.path.join(CFG.models_dir, "model_report_total_dl.json")

    payload = {
        "type": "torch_moe_v1",
        "target": "TOTAL",
        "feature_cols": feature_cols,
        "scaler": {"mean": scaler.mean_.tolist(), "std": scaler.std_.tolist()},
        "rad_col": RAD_COL,
        "ensemble": int(len(models)),
        "model_state_dicts": [m.state_dict() for m in models],
        "model_hparams": {
            "hidden": list(hidden),
            "dropout": float(args.dropout),
            "init_thr": float(init_thr),
            "init_scale": float(init_scale),
        },
        "learned_gate_thr_median": float(thr_med),
        "learned_gate_thr_all": [float(x) for x in thrs],
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
        "train": {
            "device": str(device),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "seed_base": int(args.seed),
            "num_workers": int(args.num_workers),
            "torch_threads": int(args.torch_threads),
            "torch_interop_threads": int(args.torch_interop_threads),
        },
    }

    torch.save(payload, out_model)

    report = {
        "target": "TOTAL",
        "rows_used": int(len(data)),
        "holdout_days": int(holdout),
        "calib_days": int(calib_days),
        "device": str(device),
        "ensemble": int(len(models)),
        "thr_median": float(thr_med),
        "calib_MAE_raw_kwh": float(calib_mae_raw),
        "calib_MAE_cal_kwh": float(calib_mae_cal),
        "holdout_MAE_raw_kwh": float(holdout_mae_raw),
        "holdout_MAE_cal_kwh": float(holdout_mae_cal),
        "holdout_MAE_best_kwh": float(holdout_mae_best),
        "calibration_enabled": bool(calibration_enabled),
        "feature_count": int(len(feature_cols)),
        "model_file": out_model,
        "ensemble_reports": reports,
        "train_seconds": float(time.time() - t0),
        "torch_threads": {
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
        },
        "dataloader": {"num_workers": int(args.num_workers)},
    }

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info(f"Sparade modell: {out_model}")
    log.info(f"Sparade report: {out_report}")
    log.info(
        f"Holdout: MAE_raw={holdout_mae_raw:.3f} | MAE_cal={holdout_mae_cal:.3f} | best={holdout_mae_best:.3f} | thr(med)={thr_med:.2f}"
    )
    log.info(f"Klart. Total tid: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
