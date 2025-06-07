"""
ml.py  –  simplified model-zoo (RandomForest, XGB, SVR, LSTM)
--------------------------------------------------------------
• ไม่มี logic retrain ทุก 3 เดือนอีกต่อไป  (cache เหมือนเดิม)
"""

from __future__ import annotations
import numpy as np, pandas as pd, yfinance as yf
from pathlib import Path
from joblib import dump, load

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

from config import LOOKBACK, TRAIN_SUBSET, TIMEFRAMES
from indicators import sma, rsi, macd

BASE_DIR = Path(__file__).parent  # โฟลเดอร์ที่ไฟล์ ml.py อยู่
CACHE = BASE_DIR / "models"  # …/xai/models
CACHE.mkdir(parents=True, exist_ok=True)
LOOKBACK = 90
MAX_SCORE = 40


# ───────────────────────── feature eng. ─────────────────────────
def _features(s: pd.Series, risk_norm: float = 0.0) -> pd.DataFrame:
    mom = s.pct_change(LOOKBACK)
    vol = s.pct_change().rolling(LOOKBACK).std() * np.sqrt(252)
    df = pd.concat(
        [
            mom,
            vol,
            sma(s, 14) / s - 1,
            rsi(s, 14) / 100,
            macd(s) / s,
        ],
        axis=1,
    ).dropna()
    df.columns = ["mom", "vol", "sma", "rsi", "macd"]
    df["risk"] = risk_norm  # ★ feature จากแบบสอบถาม
    df["y"] = s.pct_change().shift(-1)  # next‑day return
    return df.dropna()


# ───────────────────────── helpers ──────────────────────────
def _get_prices(tic: str, period="5y") -> pd.Series:
    df = yf.download(tic, period=period, progress=False)
    if "Close" not in df.columns:
        return pd.Series(dtype=float)
    clos = df["Close"]
    if isinstance(clos, pd.DataFrame):
        clos = clos.iloc[:, 0]
    return clos.dropna()


def _features(s: pd.Series) -> pd.DataFrame:
    mom = s.pct_change(LOOKBACK)
    vol = s.pct_change().rolling(LOOKBACK).std() * np.sqrt(252)
    df = pd.concat(
        [
            mom,
            vol,
            sma(s, 14) / s - 1,
            rsi(s, 14) / 100,
            macd(s) / s,
        ],
        axis=1,
    ).dropna()
    df.columns = ["mom", "vol", "sma", "rsi", "macd"]
    df["y"] = s.pct_change().shift(-1)
    return df.dropna()


# ───────────────────────── feature eng. ─────────────────────────
def _features(s: pd.Series, risk_norm: float = 0.0) -> pd.DataFrame:
    mom = s.pct_change(LOOKBACK)
    vol = s.pct_change().rolling(LOOKBACK).std() * np.sqrt(252)
    df = pd.concat(
        [
            mom,
            vol,
            sma(s, 14) / s - 1,
            rsi(s, 14) / 100,
            macd(s) / s,
        ],
        axis=1,
    ).dropna()
    df.columns = ["mom", "vol", "sma", "rsi", "macd"]
    df["risk"] = risk_norm  # ★ feature จากแบบสอบถาม
    df["y"] = s.pct_change().shift(-1)  # next‑day return
    return df.dropna()


# ───────────────────────── train zoo ─────────────────────────
def _train_models(X: pd.DataFrame, y: pd.Series):
    xgb = XGBRegressor(n_estimators=300, random_state=42, verbosity=0).fit(X, y)

    X3 = np.expand_dims(X.values, 1)
    lstm = Sequential(
        [
            Input((1, X.shape[1])),
            LSTM(32, activation="relu"),
            Dense(1),
        ]
    )
    lstm.compile("adam", "mse")
    lstm.fit(X3, y, epochs=50, verbose=0)
    return {"xgb": xgb, "lstm": lstm}


def train_for_timeframe(tickers: list[str], timeframe: str):
    X_parts, y_parts = [], []
    for t in tickers[:TRAIN_SUBSET]:
        s = _get_prices(t, timeframe)
        df = _features(s)
        if df.empty:
            continue
        X_parts.append(df.drop("y", axis=1))
        y_parts.append(df["y"])

    if not X_parts:
        raise RuntimeError("No training data collected")

    X = pd.concat(X_parts)
    y = pd.concat(y_parts)
    models = _train_models(X, y)

    for name, m in models.items():
        dump(m, CACHE / f"{name}_{timeframe}.joblib")
    return models


# ───────────────────────── predict view ──────────────────────
def _load(name: str, tf: str):
    return load(CACHE / f"{name}_{tf}.joblib")


def predict_view(
    tic: str,
    timeframe: str = "1y",
    risk_score: int | float = 0,
) -> tuple[float, float]:
    """Return (expected α,  1‑σ uncertainty) จากโมเดล RF/XGB/SVR/LSTM"""
    for name in ("xgb", "lstm"):
        if not (CACHE / f"{name}_{timeframe}.joblib").exists():
            from data import load_tickers

            train_for_timeframe(load_tickers(), timeframe)
            break

    prices = _get_prices(tic, timeframe)
    if len(prices) < LOOKBACK + 2:
        return np.nan, np.nan

    risk_norm = float(risk_score) / MAX_SCORE
    X = _features(prices, risk_norm).iloc[[-1]].drop("y", axis=1)

    xgb_pred = _load("xgb", timeframe).predict(X)[0]
    lstm_pred = _load("lstm", timeframe).predict(np.expand_dims(X.values, 1))[0][0]

    preds = np.array([xgb_pred, lstm_pred])
    return float(preds.mean()), float(preds.std())
