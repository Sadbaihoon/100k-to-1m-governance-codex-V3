"""Transparent analytics engine for the US Equity Research Desk.

No external data is fetched here; app_us_research.py owns data acquisition.
Designed to be compatible with current pandas versions used by Streamlit Cloud.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import pandas as pd

TRADING_DAYS = 252


def as_number(value: Any) -> float | None:
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
    if history is None or history.empty:
        return pd.Series(dtype="float64")
    for column in ("Adj Close", "Close"):
        if column in history.columns:
            return pd.to_numeric(history[column], errors="coerce").dropna()
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
    return float((prices / prices.cummax() - 1.0).min())


def _beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float | None:
    joined = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if len(joined) < 30:
        return None
    variance = float(joined.iloc[:, 1].var(ddof=1))
    if not math.isfinite(variance) or variance <= 0:
        return None
    return float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / variance)


def _slice_years(frame: pd.DataFrame, years: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    end = frame.index.max()
    start = end - pd.DateOffset(years=int(years))
    return frame.loc[frame.index >= start].copy()


def prepare_ema_chart(
    history: pd.DataFrame,
    timeframe: str,
    periods: list[int],
) -> pd.DataFrame:
    """Prepare Price + EMA for Daily, Weekly, or Monthly views.

    Uses explicit DateOffset slicing instead of DataFrame.last(), which was
    removed/changed across pandas versions.
    """
    close = _close_series(history).sort_index()
    if close.empty:
        return pd.DataFrame()

    frame = close.to_frame("Price")

    if timeframe.startswith("วัน") or "Daily" in timeframe:
        data = _slice_years(frame, 2)
    elif timeframe.startswith("สัปดาห์") or "Weekly" in timeframe:
        data = frame.resample("W-FRI").last().dropna()
        data = _slice_years(data, 5)
    else:
        # "M" is broadly compatible with pandas versions used by Streamlit.
        data = frame.resample("M").last().dropna()
        data = _slice_years(data, 20)

    if data.empty:
        return pd.DataFrame()

    for period in periods:
        period = int(period)
        if period > 0:
            data[f"EMA {period}"] = data["Price"].ewm(
                span=period, adjust=False, min_periods=period
            ).mean()

    return data


def calculate_indicators(
    history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate technical, volatility, beta and relative-performance evidence."""
    close = _close_series(history).sort_index()
    benchmark = _close_series(benchmark_history).sort_index()

    if close.empty:
        return {
            "price": None,
            "ema20": None,
            "ema50": None,
            "ema200": None,
            "rsi14": None,
            "atr14": None,
            "atr_pct": None,
            "annualized_volatility": None,
            "max_drawdown_1y": None,
            "beta": None,
            "relative_63d_return": None,
            "regime": "No data",
            "daily_returns": pd.Series(dtype="float64"),
        }

    returns = close.pct_change().dropna()

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = (100 - (100 / (1 + rs))).astype("float64")

    high = pd.to_numeric(history["High"], errors="coerce") if "High" in history else close
    low = pd.to_numeric(history["Low"], errors="coerce") if "Low" in history else close
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.rolling(14).mean()

    benchmark_returns = benchmark.pct_change().dropna()
    beta = _beta(returns, benchmark_returns)

    stock_63 = float((1 + returns.tail(63)).prod() - 1) if len(returns) >= 63 else None
    bench_63 = (
        float((1 + benchmark_returns.tail(63)).prod() - 1)
        if len(benchmark_returns) >= 63
        else None
    )
    relative = stock_63 - bench_63 if stock_63 is not None and bench_63 is not None else None

    latest = float(close.iloc[-1])
    e20, e50, e200 = float(ema20.iloc[-1]), float(ema50.iloc[-1]), float(ema200.iloc[-1])

    if latest > e20 > e50 > e200:
        regime = "Bullish"
    elif latest < e20 < e50 < e200:
        regime = "Bearish"
    else:
        regime = "Mixed / Transition"

    return {
        "price": latest,
        "ema20": e20,
        "ema50": e50,
        "ema200": e200,
        "rsi14": as_number(rsi.iloc[-1]),
        "atr14": as_number(atr14.iloc[-1]),
        "atr_pct": as_number(atr14.iloc[-1] / latest) if latest else None,
        "annualized_volatility": as_number(returns.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "max_drawdown_1y": _max_drawdown(close.tail(TRADING_DAYS)),
        "beta": beta,
        "relative_63d_return": relative,
        "regime": regime,
        "daily_returns": returns,
    }


def calculate_research_score(
    indicators: dict[str, Any],
    fundamentals: dict[str, Any],
) -> dict[str, Any]:
    """Create a transparent evidence score with explicit data coverage."""
    ema20 = as_number(indicators.get("ema20"))
    ema50 = as_number(indicators.get("ema50"))
    ema200 = as_number(indicators.get("ema200"))
    price = as_number(indicators.get("price"))
    rsi = as_number(indicators.get("rsi14"))
    relative = as_number(indicators.get("relative_63d_return"))
    volatility = as_number(indicators.get("annualized_volatility"))
    beta = as_number(indicators.get("beta"))

    fcf = as_number(fundamentals.get("free_cash_flow"))
    roe = as_number(fundamentals.get("return_on_equity"))
    margin = as_number(fundamentals.get("operating_margins"))
    growth = as_number(fundamentals.get("revenue_growth"))
    pe = as_number(fundamentals.get("trailing_pe"))
    pb = as_number(fundamentals.get("price_to_book"))

    checks = [
        ("Price > EMA20", price is not None and ema20 is not None and price > ema20, 10,
         f"{price:.2f} vs EMA20 {ema20:.2f}" if price is not None and ema20 is not None else "No data"),
        ("EMA20 > EMA50", ema20 is not None and ema50 is not None and ema20 > ema50, 10,
         f"{ema20:.2f} vs EMA50 {ema50:.2f}" if ema20 is not None and ema50 is not None else "No data"),
        ("EMA50 > EMA200", ema50 is not None and ema200 is not None and ema50 > ema200, 10,
         f"{ema50:.2f} vs EMA200 {ema200:.2f}" if ema50 is not None and ema200 is not None else "No data"),
        ("RSI available", rsi is not None, 5, f"{rsi:.1f}" if rsi is not None else "No data"),
        ("Relative 63d return positive", relative is not None and relative > 0, 10,
         f"{relative:.2%}" if relative is not None else "No data"),
        ("Volatility available", volatility is not None, 5,
         f"{volatility:.2%}" if volatility is not None else "No data"),
        ("Beta available", beta is not None, 5, f"{beta:.2f}" if beta is not None else "No data"),
        ("Free cash flow positive", fcf is not None and fcf > 0, 10,
         f"{fcf:,.0f}" if fcf is not None else "No data"),
        ("ROE positive", roe is not None and roe > 0, 5,
         f"{roe:.2%}" if roe is not None else "No data"),
        ("Operating margin positive", margin is not None and margin > 0, 5,
         f"{margin:.2%}" if margin is not None else "No data"),
        ("Revenue growth positive", growth is not None and growth > 0, 5,
         f"{growth:.2%}" if growth is not None else "No data"),
        ("Valuation data available", pe is not None or pb is not None, 5,
         f"PE={pe:.2f}, PB={pb:.2f}" if pe is not None and pb is not None
         else ("PE={:.2f}".format(pe) if pe is not None else ("PB={:.2f}".format(pb) if pb is not None else "No data"))),
    ]

    max_points = sum(item[2] for item in checks)
    earned = sum(points for _, condition, points, _ in checks if condition)
    score = earned / max_points * 100 if max_points else None

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
    coverage = sum(
        points for _, reading_available, points, reading in checks
        if reading != "No data"
    ) / max_points if max_points else 0.0

    return {"score": score, "coverage": coverage, "evidence": evidence}


def calculate_position_risk(
    daily_returns: pd.Series,
    account_value: float,
    risk_fraction: float,
    entry: float,
    stop: float,
    max_position_fraction: float,
) -> dict[str, Any]:
    account_value = max(float(account_value), 0.0)
    risk_fraction = max(float(risk_fraction), 0.0)
    entry = max(float(entry), 0.0)
    stop = max(float(stop), 0.0)
    max_position_fraction = max(float(max_position_fraction), 0.0)

    risk_budget = account_value * risk_fraction
    risk_per_share = max(entry - stop, 0.0)
    shares_by_risk = math.floor(risk_budget / risk_per_share) if risk_per_share > 0 else 0
    shares_by_cap = math.floor(
        (account_value * max_position_fraction) / entry
    ) if entry > 0 else 0
    shares = max(0, min(shares_by_risk, shares_by_cap))
    position_value = shares * entry

    returns = pd.to_numeric(daily_returns, errors="coerce").dropna().tail(TRADING_DAYS)
    if returns.empty or account_value <= 0 or position_value <= 0:
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
    """EMA20/50 long-only backtest with one-day signal shift."""
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
    equity = (1.0 + strategy_returns).cumprod()
    benchmark_equity = (1.0 + returns).cumprod()

    annual_return = _annualized_return(strategy_returns)
    volatility = (
        float(strategy_returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
        if strategy_returns.std(ddof=1) is not None
        else None
    )
    max_dd = _max_drawdown(equity)

    sharpe = (
        annual_return / volatility
        if annual_return is not None and volatility and volatility > 0
        else None
    )

    metrics = {
        "annual_return": annual_return,
        "volatility": volatility,
        "max_drawdown": max_dd,
        "benchmark_return": float(benchmark_equity.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "trades": int((turnover > 0).sum()),
    }

    chart = pd.DataFrame(
        {
            "Strategy": equity,
            "Buy & Hold": benchmark_equity,
        }
    )
    return chart, metrics
