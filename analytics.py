"""Transparent analytics engine for the US Equity Research Desk.

All calculations are deterministic and use only pandas/numpy-like pandas operations.
No external data is fetched here; app_us_research.py owns data acquisition.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import pandas as pd

TRADING_DAYS = 252


def as_number(value: Any) -> float | None:
    """Return a finite float or None for missing/non-numeric values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Real):
        number = float(value)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
    return number if math.isfinite(number) else None


def adjusted_close(history: pd.DataFrame) -> pd.Series:
    """Return adjusted close when available, otherwise close."""
    if history is None or history.empty:
        return pd.Series(dtype="float64")
    for column in ("Adj Close", "Close"):
        if column in history.columns:
            series = pd.to_numeric(history[column], errors="coerce").dropna()
            return series
    return pd.Series(dtype="float64")


def _close_series(history: pd.DataFrame) -> pd.Series:
    return adjusted_close(history)


def _annualized_return(returns: pd.Series) -> float | None:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return None
    growth = float((1.0 + returns).prod())
    years = len(returns) / TRADING_DAYS
    if years <= 0 or growth <= 0:
        return None
    return growth ** (1.0 / years) - 1.0


def _max_drawdown(prices: pd.Series) -> float | None:
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    if prices.empty:
        return None
    drawdown = prices / prices.cummax() - 1.0
    return float(drawdown.min())


def _beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float | None:
    joined = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if len(joined) < 30:
        return None
    stock = joined.iloc[:, 0]
    benchmark = joined.iloc[:, 1]
    variance = float(benchmark.var(ddof=1))
    if not math.isfinite(variance) or variance <= 0:
        return None
    return float(stock.cov(benchmark) / variance)


def prepare_ema_chart(
    history: pd.DataFrame,
    timeframe: str,
    periods: list[int],
) -> pd.DataFrame:
    """Prepare a price + EMA chart for Daily/Weekly/Monthly views."""
    close = _close_series(history)
    if close.empty:
        return pd.DataFrame()

    frame = close.to_frame("Price").sort_index()
    if timeframe.startswith("วัน") or "Daily" in timeframe:
        data = frame.last("2Y")
    elif timeframe.startswith("สัปดาห์") or "Weekly" in timeframe:
        data = frame.resample("W").last().dropna().last("5Y")
    else:
        data = frame.resample("ME").last().dropna().last("20Y")

    if data.empty:
        return pd.DataFrame()

    for period in periods:
        try:
            period_int = int(period)
        except (TypeError, ValueError):
            continue
        if period_int > 0:
            data[f"EMA {period_int}"] = data["Price"].ewm(
                span=period_int, adjust=False, min_periods=period_int
            ).mean()
    return data


def calculate_indicators(
    history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate market, momentum, risk and relative-performance indicators."""
    close = _close_series(history)
    benchmark_close = _close_series(benchmark_history)

    if close.empty:
        raise ValueError("No usable close-price history.")

    close = close.sort_index()
    returns = close.pct_change().dropna()
    benchmark_close = benchmark_close.sort_index()
    benchmark_returns = benchmark_close.pct_change().dropna()

    latest_price = float(close.iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    rsi_value = as_number(rsi)
    if rsi_value is None and float(avg_loss.iloc[-1] or 0) == 0:
        rsi_value = 100.0

    high = pd.to_numeric(history["High"], errors="coerce") if "High" in history.columns else close
    low = pd.to_numeric(history["Low"], errors="coerce") if "Low" in history.columns else close
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().iloc[-1]
    atr_value = as_number(atr14)

    recent_returns = returns.tail(TRADING_DAYS)
    annual_volatility = as_number(recent_returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
    max_dd_1y = _max_drawdown(close.tail(TRADING_DAYS + 1))
    beta_1y = _beta(returns.tail(TRADING_DAYS), benchmark_returns.tail(TRADING_DAYS))

    relative_return_63d = None
    stock_63 = close.pct_change(63).iloc[-1] if len(close) > 63 else None
    bench_63 = benchmark_close.pct_change(63).iloc[-1] if len(benchmark_close) > 63 else None
    if as_number(stock_63) is not None and as_number(bench_63) is not None:
        relative_return_63d = float(stock_63 - bench_63)

    last_date = pd.Timestamp(close.index[-1]).isoformat()

    if latest_price > ema200 and ema50 > ema200:
        regime = "Bullish"
    elif latest_price < ema200 and ema50 < ema200:
        regime = "Bearish"
    else:
        regime = "Transitional"

    return {
        "price": latest_price,
        "ema_20": ema20,
        "ema_50": ema50,
        "ema_200": ema200,
        "rsi_14": rsi_value,
        "atr_14": atr_value,
        "atr_percent": (atr_value / latest_price) if atr_value is not None and latest_price else None,
        "annual_volatility": annual_volatility,
        "max_drawdown_1y": max_dd_1y,
        "beta_1y": beta_1y,
        "relative_return_63d": relative_return_63d,
        "regime": regime,
        "last_date": last_date,
        "daily_returns": returns,
    }


def calculate_research_score(
    indicators: dict[str, Any],
    fundamentals: dict[str, Any],
) -> dict[str, Any]:
    """Build an evidence checklist score from observable, non-predictive criteria."""
    checks: list[tuple[str, bool, int, str]] = []

    def add(name: str, condition: bool, points: int, reading: str) -> None:
        checks.append((name, bool(condition), points, reading))

    price = as_number(indicators.get("price"))
    ema20 = as_number(indicators.get("ema_20"))
    ema50 = as_number(indicators.get("ema_50"))
    ema200 = as_number(indicators.get("ema_200"))
    rsi = as_number(indicators.get("rsi_14"))
    rel = as_number(indicators.get("relative_return_63d"))
    vol = as_number(indicators.get("annual_volatility"))
    beta = as_number(indicators.get("beta_1y"))
    pe = as_number(fundamentals.get("trailingPE"))
    pb = as_number(fundamentals.get("priceToBook"))
    fcf = as_number(fundamentals.get("freeCashflow"))
    roe = as_number(fundamentals.get("returnOnEquity"))
    margin = as_number(fundamentals.get("operatingMargins"))
    growth = as_number(fundamentals.get("revenueGrowth"))

    add("Price above EMA20", price is not None and ema20 is not None, 10,
        "Yes" if price is not None and ema20 is not None and price > ema20 else "No/insufficient data")
    add("EMA20 above EMA50", ema20 is not None and ema50 is not None, 10,
        "Yes" if ema20 is not None and ema50 is not None and ema20 > ema50 else "No/insufficient data")
    add("Price above EMA200", price is not None and ema200 is not None, 10,
        "Yes" if price is not None and ema200 is not None and price > ema200 else "No/insufficient data")
    add("RSI available", rsi is not None, 5, f"{rsi:.1f}" if rsi is not None else "No data")
    add("Relative 63D performance available", rel is not None, 5,
        f"{rel:.1%}" if rel is not None else "No data")
    add("Volatility available", vol is not None, 5,
        f"{vol:.1%}" if vol is not None else "No data")
    add("Beta available", beta is not None, 5,
        f"{beta:.2f}" if beta is not None else "No data")
    add("Free cash flow available", fcf is not None, 10,
        f"${fcf:,.0f}" if fcf is not None else "No data")
    add("ROE available", roe is not None, 5,
        f"{roe:.1%}" if roe is not None else "No data")
    add("Operating margin available", margin is not None, 5,
        f"{margin:.1%}" if margin is not None else "No data")
    add("Revenue growth available", growth is not None, 5,
        f"{growth:.1%}" if growth is not None else "No data")
    add("Valuation data available", pe is not None or pb is not None, 5,
        "Available" if pe is not None or pb is not None else "No data")

    max_points = sum(item[2] for item in checks)
    earned = sum(points for _, condition, points, _ in checks if condition)
    score = (earned / max_points * 100) if max_points else None
    evidence = pd.DataFrame(
        [
            {
                "Criterion": name,
                "Observed": reading,
                "Points": points if condition else 0,
                "Max points": points,
            }
            for name, condition, points, reading in checks
        ]
    )
    coverage = earned / max_points if max_points else 0.0
    return {"score": score, "coverage": coverage, "evidence": evidence}


def calculate_position_risk(
    daily_returns: pd.Series,
    account_value: float,
    risk_fraction: float,
    entry: float,
    stop: float,
    max_position_fraction: float,
) -> dict[str, Any]:
    """Calculate position size and historical 95% VaR/CVaR in dollars."""
    account_value = max(float(account_value), 0.0)
    risk_fraction = max(float(risk_fraction), 0.0)
    entry = max(float(entry), 0.0)
    stop = max(float(stop), 0.0)
    max_position_fraction = max(float(max_position_fraction), 0.0)

    risk_budget = account_value * risk_fraction
    risk_per_share = max(entry - stop, 0.0)
    shares_by_risk = math.floor(risk_budget / risk_per_share) if risk_per_share > 0 else 0
    shares_by_cap = math.floor((account_value * max_position_fraction) / entry) if entry > 0 else 0
    shares = max(0, min(shares_by_risk, shares_by_cap))
    position_value = shares * entry

    returns = pd.to_numeric(daily_returns, errors="coerce").dropna().tail(TRADING_DAYS)
    if returns.empty or account_value <= 0:
        var_95 = None
        cvar_95 = None
    else:
        loss_distribution = -returns * position_value
        var_95 = float(loss_distribution.quantile(0.95))
        tail = loss_distribution[loss_distribution >= var_95]
        cvar_95 = float(tail.mean()) if not tail.empty else var_95

    return {
        "risk_budget": risk_budget,
        "risk_per_share": risk_per_share,
        "shares": shares,
        "position_value": position_value,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


def run_ema_backtest(
    history: pd.DataFrame,
    years: int,
    cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backtest an EMA20/50 trend filter with one-day signal shift.

    Long when EMA20 > EMA50; otherwise flat. Transaction costs are charged
    when position changes. The signal is shifted one trading day to prevent
    look-ahead bias.
    """
    close = _close_series(history).sort_index()
    if close.empty:
        return pd.DataFrame(), {}

    cutoff = close.index[-1] - pd.DateOffset(years=int(years))
    close = close.loc[close.index >= cutoff]
    if len(close) < 252:
        return pd.DataFrame(), {}

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    raw_signal = (ema20 > ema50).astype(float)
    signal = raw_signal.shift(1).fillna(0.0)

    returns = close.pct_change().fillna(0.0)
    turnover = signal.diff().abs().fillna(signal.abs())
    cost = float(cost_bps) / 10000.0
    strategy_returns = signal * returns - turnover * cost

    growth = (1.0 + strategy_returns).cumprod()
    benchmark_growth = (1.0 + returns).cumprod()

    annual_return = _annualized_return(strategy_returns)
    annual_volatility = as_number(strategy_returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
    max_drawdown = _max_drawdown(growth)
    benchmark_return = _annualized_return(returns)
    trades = int(turnover.sum())

    sharpe = None
    if strategy_returns.std(ddof=1) and math.isfinite(float(strategy_returns.std(ddof=1))):
        sharpe = float(strategy_returns.mean() / strategy_returns.std(ddof=1) * math.sqrt(TRADING_DAYS))

    chart = pd.DataFrame(
        {
            "Strategy": growth,
            "Buy & Hold": benchmark_growth,
        }
    )

    metrics = {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown,
        "benchmark_return": benchmark_return,
        "sharpe": sharpe,
        "trades": trades,
    }
    return chart, metrics
