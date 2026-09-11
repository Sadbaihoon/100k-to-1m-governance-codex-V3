"""
scoring_engine.py — V3 Investment Governance scoring layer.

Research Evidence Score = evidence strength, NOT a Buy Rating.
Risk Score = risk severity; higher = higher risk.
Early Warning = emerging-problem status.
Investment Decision = rule-based governance outcome.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, List
import math

def _num(v):
    try:
        if v is None or isinstance(v, bool): return None
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, float(x)))

def _avg(vals, default=50.0):
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else default

# ---------- Research Evidence ----------

def fundamental_evidence_score(f):
    keys = ["revenueGrowth","operatingMargins","freeCashflow",
            "returnOnEquity","trailingEps","totalDebt"]
    return 100 * sum(_num(f.get(k)) is not None for k in keys) / len(keys)

def technical_evidence_score(i):
    keys = ["ema20","ema50","ema200","rsi14"]
    return 100 * sum(_num(i.get(k)) is not None for k in keys) / len(keys)

def relative_performance_score(i):
    keys = ["relativeReturn63d","beta"]
    return 100 * sum(_num(i.get(k)) is not None for k in keys) / len(keys)

def valuation_evidence_score(f):
    keys = ["trailingPE","priceToBook","freeCashflow"]
    return 100 * sum(_num(f.get(k)) is not None for k in keys) / len(keys)

def risk_evidence_score(i):
    keys = ["volatility","maxDrawdown","beta"]
    return 100 * sum(_num(i.get(k)) is not None for k in keys) / len(keys)

def data_quality_score(i, f):
    return _avg([
        fundamental_evidence_score(f),
        100.0 if _num(i.get("beta")) is not None else 0.0,
        100.0 if i else 0.0
    ])

def calculate_research_evidence_score(indicators, fundamentals):
    components = {
        "fundamental_evidence": fundamental_evidence_score(fundamentals),
        "technical_evidence": technical_evidence_score(indicators),
        "relative_performance": relative_performance_score(indicators),
        "valuation_evidence": valuation_evidence_score(fundamentals),
        "risk_evidence": risk_evidence_score(indicators),
        "data_quality": data_quality_score(indicators, fundamentals),
    }
    weights = {
        "fundamental_evidence": .30, "technical_evidence": .20,
        "relative_performance": .10, "valuation_evidence": .10,
        "risk_evidence": .15, "data_quality": .15
    }
    score = sum(components[k]*weights[k] for k in weights)
    return {"score": round(_clamp(score),1),
            "label":"Research Evidence Score",
            "components":{k:round(v,1) for k,v in components.items()},
            "weights":weights,
            "interpretation":"Evidence strength and data coverage. Not a Buy Rating."}

# ---------- Risk ----------

def volatility_risk(v):
    x=_num(v)
    if x is None: return 50
    if x<=20:return 10
    if x<=30:return 25
    if x<=45:return 45
    if x<=60:return 65
    if x<=80:return 80
    return 95

def beta_risk(v):
    x=_num(v)
    if x is None:return 50
    if x<.8:return 15
    if x<=1.2:return 30
    if x<=1.5:return 50
    if x<=2:return 70
    if x<=3:return 85
    return 95

def drawdown_risk(v):
    x=_num(v)
    if x is None:return 50
    p=abs(x*100) if abs(x)<=1 else abs(x)
    if p<=10:return 10
    if p<=20:return 25
    if p<=30:return 45
    if p<=40:return 65
    if p<=50:return 80
    return 95

def fundamental_risk(f):
    r=[]
    x=_num(f.get("revenueGrowth"))
    if x is not None:r.append(15 if x>=.20 else 30 if x>=.10 else 50 if x>=0 else 75 if x>=-.10 else 90)
    x=_num(f.get("operatingMargins"))
    if x is not None:r.append(20 if x>=.20 else 35 if x>=.10 else 55 if x>=0 else 80)
    x=_num(f.get("freeCashflow"))
    if x is not None:r.append(20 if x>0 else 80)
    x=_num(f.get("returnOnEquity"))
    if x is not None:r.append(20 if x>=.15 else 40 if x>=.08 else 60 if x>=0 else 80)
    return _avg(r)

def valuation_risk(f):
    r=[]
    x=_num(f.get("trailingPE"))
    if x is not None and x>0:r.append(20 if x<=20 else 40 if x<=30 else 60 if x<=50 else 80 if x<=100 else 95)
    x=_num(f.get("priceToBook"))
    if x is not None and x>0:r.append(20 if x<=2 else 40 if x<=4 else 60 if x<=7 else 80 if x<=12 else 95)
    return _avg(r)

def event_risk(level=0):
    return {0:10,1:40,2:70,3:90}.get(int(level),50)

def position_risk(loss_pct):
    x=_num(loss_pct)
    if x is None:return 50
    p=abs(x)
    if p<=1:return 10
    if p<=2:return 30
    if p<=3:return 50
    if p<=5:return 75
    return 95

def calculate_risk_score(indicators, fundamentals, *, event_level=0, position_loss_pct=None):
    components={
        "volatility_risk":volatility_risk(indicators.get("volatility")),
        "beta_risk":beta_risk(indicators.get("beta")),
        "drawdown_risk":drawdown_risk(indicators.get("maxDrawdown")),
        "fundamental_risk":fundamental_risk(fundamentals),
        "valuation_risk":valuation_risk(fundamentals),
        "event_risk":event_risk(event_level),
        "position_risk":position_risk(position_loss_pct)}
    weights={"volatility_risk":.20,"beta_risk":.15,"drawdown_risk":.15,
             "fundamental_risk":.15,"valuation_risk":.15,
             "event_risk":.10,"position_risk":.10}
    score=sum(components[k]*weights[k] for k in weights)
    return {"score":round(_clamp(score),1),"label":"Risk Score",
            "direction":"Higher = Higher Risk",
            "components":{k:round(v,1) for k,v in components.items()},
            "weights":weights}

# ---------- Early Warning + Decision ----------

def evaluate_early_warning(*, risk_score, critical_signals=None, warning_signals=None):
    critical_signals=critical_signals or []
    warning_signals=warning_signals or []
    if critical_signals or risk_score>=80: status="CRITICAL"
    elif risk_score>=60 or len(warning_signals)>=2: status="WARNING"
    elif risk_score>=40 or warning_signals: status="WATCH"
    else: status="NORMAL"
    return {"status":status,"critical_signals":critical_signals,
            "warning_signals":warning_signals,
            "override":bool(critical_signals) or risk_score>=80}

def determine_investment_decision(*, research_score, risk_score,
                                  early_warning_status, thesis_status="INTACT",
                                  market_regime="NEUTRAL"):
    ew=early_warning_status.upper()
    thesis=thesis_status.upper()
    regime=market_regime.upper()
    if thesis=="INVALIDATED":
        decision,reason="EXIT REVIEW","Investment thesis is invalidated."
    elif ew=="CRITICAL" or risk_score>=80:
        decision,reason="DO NOT INITIATE","Critical risk requires a governance stop."
    elif ew=="WARNING" or risk_score>=60:
        decision,reason="HOLD / WAIT","Elevated risk requires further review."
    elif ew=="WATCH" or risk_score>=40:
        decision,reason="WATCH","Risk or emerging-warning conditions require monitoring."
    elif research_score>=75 and thesis=="INTACT" and regime!="BEARISH":
        decision,reason="ELIGIBLE FOR REVIEW","Evidence is strong and no major governance block is active."
    else:
        decision,reason="WATCH","Evidence is not sufficient for active review."
    return {"decision":decision,"reason":reason,
            "governance_principle":"No single score can authorize an investment."}

def build_governance_snapshot(indicators, fundamentals, *,
                              event_level=0, position_loss_pct=None,
                              critical_signals=None, warning_signals=None,
                              thesis_status="INTACT", market_regime="NEUTRAL"):
    evidence=calculate_research_evidence_score(indicators,fundamentals)
    risk=calculate_risk_score(indicators,fundamentals,event_level=event_level,
                              position_loss_pct=position_loss_pct)
    early=evaluate_early_warning(risk_score=risk["score"],
                                  critical_signals=critical_signals,
                                  warning_signals=warning_signals)
    decision=determine_investment_decision(
        research_score=evidence["score"], risk_score=risk["score"],
        early_warning_status=early["status"], thesis_status=thesis_status,
        market_regime=market_regime)
    return {"research_evidence":evidence,"risk":risk,
            "early_warning":early,"decision":decision}
