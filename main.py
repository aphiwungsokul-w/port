from __future__ import annotations

"""Main FastAPI entry‑point – Black‑Litterman + ML Robo‑Advisor (robust).

🔄 2025‑05‑20
    • volatility filter per risk_mode (previous commit)
    • robust optimiser – graceful fallback when CVXPY reports infeasible:
        1. Sharpe max → MinRisk → Equal‑weight fallback
        2. Automatically relax sector caps / soft cap isn’t needed here – equal weight always feasible
"""

import hashlib
import json
from typing import Dict, List, Tuple
import re
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import fundamental as fun

# project modules
import bl  # Black‑Litterman helper
from bl import _make_cov_pd  # PD‑fix helper
import data
import ml
import plotting_utils as pu

# ───────────────────────── Questionnaire scoring ──────────────────────────
BANDS = [
    (0, 14, 1, "เสี่ยงต่ำ"),
    (15, 21, 2, "เสี่ยงปานกลางค่อนข้างต่ำ"),
    (22, 29, 3, "เสี่ยงปานกลางค่อนข้างสูง"),
    (30, 36, 4, "เสี่ยงสูง"),
    (37, 40, 5, "เสี่ยงสูงมาก"),
]
LAMBDA_MAP = {1: 10.0, 2: 6.0, 3: 4.0, 4: 2.5, 5: 1.2}

def score_questionnaire(raw: str):
    """
    รับสตริงเช่น  '1,4,3,1|3|5,...'  → คืน
    (total_score, level 1-5, lambda_ra, explanation_str)
    """
    nums = list(map(int, re.findall(r'\d+', raw)))   # ดึงทุกเลข
    pts  = sum(nums)                                 # 10–40

    for lo, hi, lvl, desc in BANDS:
        if lo <= pts <= hi:
            expl = f"คุณได้ {pts} คะแนน → ระดับ {lvl} ({desc})"
            return pts, lvl, LAMBDA_MAP[lvl], expl

    # fallback (กรณีคะแนนนอกขอบเขต)
    return pts, 3, LAMBDA_MAP[3], "ไม่พบคะแนนแบบสอบถาม"


# ───────────────────────── Risk parameters ────────────────────────────────
SECTOR_CAP = 0.25  # 25 % per sector
RISK_PARAM = {
    # level 1 – เสี่ยงต่ำมาก
    "level1": {"soft": 0.02, "hard": 0.06, "vol": 0.20},
    # level 2 – เสี่ยงปานกลางค่อนข้างต่ำ
    "level2": {"soft": 0.04, "hard": 0.08, "vol": 0.30},
    # level 3 – เสี่ยงปานกลางค่อนข้างสูง
    "level3": {"soft": 0.06, "hard": 0.10, "vol": 0.40},
    # level 4 – เสี่ยงสูง
    "level4": {"soft": 0.08, "hard": 0.12, "vol": 0.50},
    # level 5 – เสี่ยงสูงมาก
    "level5": {"soft": 0.10, "hard": 0.15, "vol": 0.60},
}

TF_ALIAS = {"short": "3mo", "medium": "1y", "long": "5y"}
TF_CFG: Dict[str, Dict[str, float]] = {
    "3mo": {"boost": 0.90, "ann": 0.25},
    "1y": {"boost": 1.00, "ann": 1.00},
    "5y": {"boost": 1.50, "ann": 5.00},
}

# ───────────────────────── FastAPI init ───────────────────────────────────
app = FastAPI(title="Robo‑Advisor BL+ML API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates("templates")

# ───────────────────────── Helpers ────────────────────────────────────────


def _user_offset(uid: str, tic: str) -> float:
    h = hashlib.md5(f"{uid}_{tic}".encode()).digest()
    return (int.from_bytes(h[:4], "little") / 2**32 - 0.5) * 0.05


def _sector_of(tic: str) -> str:
    try:
        return yf.Ticker(tic).info.get("sector", "Unknown") or "Unknown"
    except Exception:
        return "Unknown"


# ───────────────────────── Core route ─────────────────────────────────────


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/portfolio")
def portfolio(
    user_id: str = Query("guest"),
    capital: float = Query(100_000, ge=0),
    timeframe: str = Query("1y"),
    questionnaire: str = Query(""),
):
    # 1) Risk params
    risk_score, risk_level, lambda_ra, explanation = score_questionnaire(questionnaire)
    risk_key = f"level{risk_level}"
    params = RISK_PARAM[risk_key]
    soft_cap, hard_cap, max_vol = params["soft"], params["hard"], params["vol"]

    # 2) Timeframe mapping
    timeframe = TF_ALIAS.get(timeframe, timeframe)
    cfg = TF_CFG.get(timeframe, TF_CFG["1y"])

    # 3) Universe
    raw = data.load_tickers()
    tickers = fun.screen_universe(raw)
    if len(tickers) < 10:  # กันกรณีข้อมูลไม่ครบ
        tickers = raw
    sector_map = {t: _sector_of(t) for t in tickers}

    # 4) Prices & volatility filter
    price = yf.download(tickers, period=timeframe, progress=False)["Close"].ffill()
    if price.empty:
        return Response("{}", media_type="application/json")
    vol = price.pct_change().std() * np.sqrt(252)
    tickers = [t for t in tickers if vol[t] <= max_vol]
    if len(tickers) < 5:
        tickers = vol.sort_values().index[:20].tolist()  # fallback smallest vol
    price = price[tickers]
    sector_map = {t: sector_map[t] for t in tickers}

    # 5) Market returns
    market_mu = {}
    for t in tickers:
        s = price[t].dropna()
        if len(s) < 2:
            continue
        r_tot = s.iloc[-1] / s.iloc[0] - 1
        market_mu[t] = (1 + r_tot) ** (1 / cfg["ann"]) - 1
    if not market_mu:
        return Response("{}", media_type="application/json")

    # 6) ML views
    ml_tf = "1y" if timeframe == "3mo" else timeframe
    views = {}
    for t in market_mu:
        try:
            v, u = ml.predict_view(t, timeframe=ml_tf, risk_score=risk_score)
        except RuntimeError:
            v, u = 0.0, 0.3
        views[t] = (v * cfg["boost"] + _user_offset(user_id, t), u)

    # 7) Build BL portfolio
    port = bl.build_bl_port(market_mu, views, lambda_ra, sector_map, timeframe)
    port.cov = _make_cov_pd(port.cov)
    port.upperlng = soft_cap

    # ---------------- sector inequality caps ------------------
    # ดึงชื่อสินทรัพย์จาก DataFrame / Series ใด ๆ
    assets = (
        list(port.mu.columns)
        if isinstance(port.mu, pd.DataFrame)
        else list(port.mu.index)
    )
    n = len(assets)

    A, b = [], []
    for sec in {sector_map[t] for t in assets}:
        idx = [i for i, t in enumerate(assets) if sector_map[t] == sec]
        row = np.zeros(n)
        row[idx] = 1
        A.append(row)
        b.append(SECTOR_CAP)

    if A:  # เซ็ต constraint เข้า Riskfolio
        port.ainequality = np.asarray(A)
        port.binequality = np.asarray(b).reshape(-1, 1)

    # 8) Optimisation attempts
    def _optim(obj: str):
        try:
            return port.optimization(
                model="Classic", rm="CVaR", obj=obj, rf=0.02, l=lambda_ra
            )
        except Exception:
            return None

    w_df = _optim("Sharpe")
    if w_df is None or (hasattr(w_df, "empty") and w_df.empty):
        w_df = _optim("MinRisk")

    # 9) Fallback equal‑weight if optimiser infeasible/empty
    if w_df is None or w_df.empty:
        ew = pd.Series(1 / len(market_mu), index=list(market_mu))
        weights = ew
    else:
        weights = w_df["weights"].clip(lower=0, upper=hard_cap)
        weights = weights[weights > 0]
        weights /= weights.sum()

    # 10) Prepare response
    top = weights.sort_values(ascending=False).head(20)

    tickers_in_port = list(weights.index)  # ชื่อหุ้นที่เหลือหลัง optimise

    #   (a) Covariance / Correlation Heat-map
    cov_fig = pu.cov_heatmap(
        port.cov.loc[tickers_in_port, tickers_in_port],  # matrix ตัดเฉพาะหุ้นที่ใช้
        corr=False,  # = Covariance Heat-map
    )

    #   (b) Efficient Frontier  (สุ่มพอร์ต + จุด frontier)
    ef_fig, _ = pu.efficient_frontier(
        port, tickers=tickers_in_port, points=300  # ปรับจำนวน random points ได้
    )

    plots = {
        "weights": json.loads(pu.weights_bar(top.to_dict()).to_json()),
        "cov": json.loads(cov_fig.to_json()),  # ← เพิ่ม
        "ef": json.loads(ef_fig.to_json()),  # ← เพิ่ม
    }
    # ── data สำหรับ Google Pie ───────
    chart_data = [{"label": t, "value": round(w * 100, 4)} for t, w in top.items()]

    # ── ตาราง top-20 ─────────────────
    top20 = [
        {
            "ticker": t,
            "weight": round(w, 4),
            "money": round(w * capital, 2),
            "industry": sector_map.get(t, "Unknown"),  # ← ชื่อนี้ตรงกับ front-end
        }
        for t, w in top.items()
    ]

    return JSONResponse(
        {
            "user_id": user_id,
            "capital": capital,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_explantion": explanation,
            "lambda": lambda_ra,
            "vol_max": max_vol,
            "top20": top20,
            "chart_data": chart_data,
            "plots": plots,
        }
    )


# ── Local dev run ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8000)
