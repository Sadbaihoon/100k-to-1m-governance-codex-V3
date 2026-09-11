"""Transparent, reproducible analytics for the US equity research dashboard.

These calculations are decision-support metrics, not predictions or investment
recommendations. Every function accepts plain market history so it can later be
fed by a licensed data provider without changing the app interface.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import pandas as pd


TRADING_DAYS = 252


def as_number(value: object) -> float | None:
    """Convert an API value to float without treating missing data as zero."""
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def adjusted_close(history: pd.DataFrame) -> pd.Series:
    """Use adjusted close for return and risk calculations when supplied."""
    column = "Adj Close" if "Adj Close" in history.columns else "Close"
    return history[column].dropna().astype(float)


def prepare_ema_chart(history: pd.DataFrame, timeframe: str, periods: list[int]) -> pd.DataFrame:
    """Resample adjusted prices and calculate selected EMA lines."""
    if history.empty:
        return pd.DataFrame()

    close = adjusted_close(history)
    if timeframe == "วัน (Daily)":
        series = close
        display_series = series.loc[series.index >= series.index.max() - pd.DateOffset(years=2)]
    elif timeframe == "สัปดาห์ (Weekly)":
        series = close.resample("W-FRI").last().dropna()
        display_series = series.loc[series.index >= series.index.max() - pd.DateOffset(years=5)]
    else:
        series = close.resample("MS").last().dropna()
        display_series = series.loc[series.index >= series.index.max() - pd.DateOffset(years=20)]

    chart = pd.DataFrame({"Adjusted price": display_series})
    for period in periods:
        chart[f"EMA {period}"] = series.ewm(
            span=period, adjust=False, min_periods=period
        ).mean().reindex(display_series.index)
    return chart


def _annualized_return(returns: pd.Series) -> float | None:
    returns = returns.dropna()
    if len(returns) < 2:
        return None
    total_return = float((1 + returns).prod())
    if total_return <= 0:
        return None
    return total_return ** (TRADING_DAYS / len(returns)) - 1


def _max_drawdown(returns: pd.Series) -> float | None:
    returns = returns.dropna()
    if returns.empty:
        return None
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())


def calculate_indicators(history: pd.DataFrame, benchmark_history: pd.DataFrame) -> dict[str, Any]:
    """Calculate transparent technical and risk metrics from daily OHLCV data."""
    prices = adjusted_close(history)
    returns = prices.pct_change().dropna()
    lookback_returns = returns.tail(TRADING_DAYS)

    ema_20 = prices.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_50 = prices.ewm(span=50, adjust=False, min_periods=50).mean()
    ema_200 = prices.ewm(span=200, adjust=False, min_periods=200).mean()

    delta = prices.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = gains / losses.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + relative_strength))

    previous_close = history["Close"].shift(1)
    true_range = pd.concat(
        [
            history["High"] - history["Low"],
            (history["High"] - previous_close).abs(),
            (history["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()

    benchmark_returns = adjusted_close(benchmark_history).pct_change().dropna()
    aligned_returns = pd.concat(
        [returns.rename("stock"), benchmark_returns.rename("benchmark")], axis=1, join="inner"
    ).dropna().tail(TRADING_DAYS)
    beta: float | None = None
    if len(aligned_returns) >= 60 and aligned_returns["benchmark"].var() > 0:
        beta = float(aligned_returns["stock"].cov(aligned_returns["benchmark"]) / aligned_returns["benchmark"].var())

    price = float(prices.iloc[-1])
    last_ema_20 = as_number(ema_20.iloc[-1])
    last_ema_50 = as_number(ema_50.iloc[-1])
    last_ema_200 = as_number(ema_200.iloc[-1])
    annual_volatility = (
        float(lookback_returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
        if len(lookback_returns) >= 20
        else None
    )
    max_drawdown = _max_drawdown(lookback_returns)
    atr_value = as_number(atr.iloc[-1])

    if last_ema_50 is not None and last_ema_200 is not None and price > last_ema_50 > last_ema_200:
        regime = "Bullish trend"
    elif last_ema_50 is not None and last_ema_200 is not None and price < last_ema_50 < last_ema_200:
        regime = "Bearish trend"
    else:
        regime = "Mixed / transition"

    relative_63d: float | None = None
    if len(aligned_returns) >= 63:
        stock_return = (1 + aligned_returns["stock"].tail(63)).prod() - 1
        benchmark_return = (1 + aligned_returns["benchmark"].tail(63)).prod() - 1
        relative_63d = float(stock_return - benchmark_return)

    return {
        "price": price,
        "last_date": prices.index[-1].date().isoformat(),
        "ema_20": last_ema_20,
        "ema_50": last_ema_50,
        "ema_200": last_ema_200,
        "rsi_14": as_number(rsi.iloc[-1]),
        "atr_14": atr_value,
        "atr_percent": atr_value / price if atr_value is not None and price else None,
        "annual_volatility": annual_volatility,
        "max_drawdown_1y": max_drawdown,
        "beta_1y": beta,
        "relative_return_63d": relative_63d,
        "regime": regime,
        "daily_returns": lookback_returns,
    }


def calculate_research_score(indicators: dict[str, Any], company: dict[str, object]) -> dict[str, Any]:
    """Score observable evidence and report its data coverage, never infer gaps."""
    earned = 0.0
    available = 0.0
    evidence: list[dict[str, Any]] = []

    def assess(label: str, value: object, predicate: Any, weight: float) -> None:
        nonlocal earned, available
        if isinstance(value, tuple):
            missing = any(item is None or pd.isna(item) for item in value)
        else:
            missing = value is None or pd.isna(value)
        if missing:
            evidence.append({"Metric": label, "Result": "No data", "Weight": weight})
            return
        available += weight
        passed = bool(predicate(value))
        if passed:
            earned += weight
        evidence.append({"Metric": label, "Result": "Pass" if passed else "Review", "Weight": weight})

    price = indicators["price"]
    assess("Price above EMA 50", indicators["ema_50"], lambda value: price > value, 10)
    assess("EMA 50 above EMA 200", (indicators["ema_50"], indicators["ema_200"]), lambda value: value[0] > value[1], 12)
    assess("RSI within 40–70", indicators["rsi_14"], lambda value: 40 <= value <= 70, 6)
    assess("Annualized volatility ≤ 35%", indicators["annual_volatility"], lambda value: value <= 0.35, 8)
    assess("1-year drawdown above −30%", indicators["max_drawdown_1y"], lambda value: value >= -0.30, 8)
    assess("Positive free cash flow", company.get("freeCashflow"), lambda value: value > 0, 10)
    assess("Return on equity ≥ 15%", company.get("returnOnEquity"), lambda value: value >= 0.15, 10)
    assess("Operating margin ≥ 10%", company.get("operatingMargins"), lambda value: value >= 0.10, 8)
    assess("Positive revenue growth", company.get("revenueGrowth"), lambda value: value > 0, 8)

    score = (earned / available * 100) if available else None
    return {
        "score": score,
        "coverage": available / 80,
        "evidence": pd.DataFrame(evidence),
    }


def calculate_position_risk(
    daily_returns: pd.Series,
    capital: float,
    risk_fraction: float,
    entry: float,
    stop: float,
    max_position_fraction: float,
) -> dict[str, float | int | None]:
    """Size a position from a fixed loss budget and report historical tail risk."""
    risk_per_share = entry - stop
    if capital <= 0 or risk_fraction <= 0 or entry <= 0 or risk_per_share <= 0:
        return {"shares": 0, "position_value": 0.0, "risk_budget": 0.0, "risk_per_share": risk_per_share, "var_95": None, "cvar_95": None}

    risk_budget = capital * risk_fraction
    shares_by_risk = math.floor(risk_budget / risk_per_share)
    shares_by_cap = math.floor((capital * max_position_fraction) / entry)
    shares = max(0, min(shares_by_risk, shares_by_cap))
    position_value = shares * entry

    returns = daily_returns.dropna().tail(TRADING_DAYS)
    var_95: float | None = None
    cvar_95: float | None = None
    if len(returns) >= 60 and position_value > 0:
        cutoff = float(returns.quantile(0.05))
        var_95 = max(0.0, -cutoff * position_value)
        tail = returns[returns <= cutoff]
        if not tail.empty:
            cvar_95 = max(0.0, -float(tail.mean()) * position_value)

    return {
        "shares": shares,
        "position_value": position_value,
        "risk_budget": risk_budget,
        "risk_per_share": risk_per_share,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


def run_ema_backtest(history: pd.DataFrame, years: int, cost_bps: float) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Backtest a daily EMA(20/50) trend filter without look-ahead bias."""
    prices = adjusted_close(history)
    start = prices.index.max() - pd.DateOffset(years=years)
    prices = prices.loc[prices.index >= start]
    returns = prices.pct_change().dropna()
    if len(returns) < 252:
        return pd.DataFrame(), {"annual_return": None, "annual_volatility": None, "max_drawdown": None, "sharpe": None, "trades": 0, "benchmark_return": None}

    fast = prices.ewm(span=20, adjust=False, min_periods=20).mean()
    slow = prices.ewm(span=50, adjust=False, min_periods=50).mean()
    signal = (fast > slow).astype(float).shift(1).reindex(returns.index).fillna(0.0)
    turnover = signal.diff().abs().fillna(signal.abs())
    strategy_returns = returns * signal - turnover * (cost_bps / 10_000)
    strategy_returns = strategy_returns.dropna()
    equity = (1 + strategy_returns).cumprod()
    chart = pd.DataFrame({"EMA 20/50 strategy": equity, "Buy and hold": (1 + returns.reindex(equity.index)).cumprod()})

    annual_return = _annualized_return(strategy_returns)
    annual_volatility = float(strategy_returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
    sharpe = annual_return / annual_volatility if annual_return is not None and annual_volatility > 0 else None
    return chart, {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": _max_drawdown(strategy_returns),
        "sharpe": sharpe,
        "trades": int((turnover > 0).sum()),
        "benchmark_return": _annualized_return(returns.reindex(strategy_returns.index)),
    }
