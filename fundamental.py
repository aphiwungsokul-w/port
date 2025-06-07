# fundamental.py  – quality screener v2  (Damodaran × McKinsey)
# ------------------------------------------------------------
# ใช้ได้ทั้ง stand-alone และเรียกจาก main.py

import math
from functools import lru_cache
from typing import List

import yfinance as yf

TAX_RATE = 0.20           # อัตราภาษีนิติบุคคลไทย 20%
PASS_THRESHOLD = 6        # ต้องผ่าน ≥ 6/8

# ---------- helpers -------------------------------------------------------

@lru_cache(maxsize=1024)
def _info(tic: str) -> dict:
    t = yf.Ticker(tic)
    d = dict(getattr(t, "fast_info", {}))
    d.update(t.info or {})
    return d

def _safe_num(x, default=0.0):
    try:
        return float(x) if x not in (None, "None") else default
    except Exception:
        return default

# ---------- metric rules --------------------------------------------------

def _eps_growth_next5y(d):
    g = _safe_num(d.get("earningsGrowth"))
    if not g:
        fwd, trl = map(_safe_num, (d.get("forwardEps"), d.get("trailingEps")))
        g = (fwd - trl) / abs(trl) if trl else 0
    return g > 0

def _revenue_growth_5y(d):
    return _safe_num(d.get("revenueGrowth")) > 0.05

def _debt_to_ebitda(d):
    debt  = _safe_num(d.get("totalDebt"))
    ebitda = _safe_num(d.get("ebitda"))
    return (debt and ebitda) and debt / ebitda < 1.5

def _fcf_yield_pos(d):
    fcf, mcap = map(_safe_num, (d.get("freeCashflow"), d.get("marketCap")))
    return (fcf and mcap) and fcf / mcap > 0

def _roic_gt_8(d):
    roic = _safe_num(d.get("returnOnInvestedCapital"))
    if not roic:
        roic = _safe_num(d.get("returnOnEquity"))
    return roic > 0.08

def _sgr_pos(d):
    eps_g = _safe_num(d.get("earningsGrowth"))
    roe   = _safe_num(d.get("returnOnEquity"))
    roic  = _safe_num(d.get("returnOnInvestedCapital") or d.get("returnOnEquity"))
    sgr = eps_g * roic / roe if roe else 0
    return sgr > 0

def _ev_sales_atom(d):
    ev   = _safe_num(d.get("enterpriseValue"))
    rev  = _safe_num(d.get("totalRevenue"))
    ebit_margin = _safe_num(d.get("ebitMargins")) or _safe_num(d.get("ebitdaMargins"))
    if not (ev and rev and ebit_margin):
        return False
    ev_sales = ev / rev
    atom = ebit_margin * (1 - TAX_RATE)
    return ev_sales <= atom / 2

def _peg_vs_roic(d):
    peg  = _safe_num(d.get("pegRatio"), default=math.inf)
    roic = _safe_num(d.get("returnOnInvestedCapital") or d.get("returnOnEquity"))
    if roic >= 0.20:   return peg <= 1.5
    if 0.10 <= roic < 0.20: return peg <= 1.0
    return False

_RULES = [
    _eps_growth_next5y,     # 1
    _revenue_growth_5y,     # 2
    _debt_to_ebitda,        # 3
    _fcf_yield_pos,         # 4
    _roic_gt_8,             # 5
    _sgr_pos,               # 6
    _ev_sales_atom,         # 7
    _peg_vs_roic,           # 8
]

# ---------- public API ----------------------------------------------------

def score(ticker: str) -> float:
    """Return 0-1 = ratio of rules passed."""
    d = _info(ticker)
    passed = sum(rule(d) for rule in _RULES)
    return passed / len(_RULES)

def pass_filters(ticker: str) -> bool:
    """True หากผ่าน ≥ PASS_THRESHOLD เกณฑ์"""
    d = _info(ticker)
    return sum(rule(d) for rule in _RULES) >= PASS_THRESHOLD

def screen_universe(tickers: List[str]) -> List[str]:
    return [t for t in tickers if pass_filters(t)]
