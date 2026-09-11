"""US equity research dashboard with transparent, institutional-style controls.

Run with: python -m streamlit run app_us_research.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from html import escape
from typing import Any
from urllib.parse import quote

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

from analytics import (
    adjusted_close,
    as_number,
    calculate_indicators,
    calculate_position_risk,
    calculate_research_score,
    prepare_ema_chart,
    run_ema_backtest,
)
from storage import get_decisions, initialize_database, save_decision


BENCHMARK = "^GSPC"
TRADINGVIEW_EXCHANGE_MAP = {
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NYQ": "NYSE",
    "ASE": "AMEX",
}
st.set_page_config(page_title="US Equity Research Desk", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
        .block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 3rem; }
        .hero { padding: 1.5rem 1.7rem; border-radius: 1rem; color: white;
          background: linear-gradient(120deg, #102a43, #1f5f70); margin-bottom: 1.5rem; }
        .hero h1 { color: white; margin: 0 0 .25rem; font-size: 2rem; }
        .hero p { margin: 0; opacity: .93; }
        .section-kicker { color: #1f5f70; font-size: .8rem; font-weight: 700; letter-spacing: .08em; }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_money(value: float | None, decimals: int = 2) -> str:
    return f"${value:,.{decimals}f}" if value is not None else "No data"


def format_percent(value: float | None, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%" if value is not None else "No data"


def format_multiple(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "No data"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_us_ticker(raw_ticker: str) -> str:
    """Keep a ticker safe for display URLs and third-party widget configuration."""
    return "".join(character for character in raw_ticker.upper().strip() if character.isalnum() or character in ".-")


def cooling_key(ticker: str, minutes: int) -> str:
    return f"us_research_cooling_{ticker}_{minutes}"


def cooling_status(ticker: str, minutes: int) -> tuple[bool, str]:
    started_at = st.session_state.get(cooling_key(ticker, minutes))
    if not started_at:
        return False, "Not started"
    remaining = started_at + timedelta(minutes=minutes) - utc_now()
    if remaining.total_seconds() <= 0:
        return True, "Complete"
    minutes_left, seconds_left = divmod(int(remaining.total_seconds()), 60)
    return False, f"{minutes_left}m {seconds_left}s remaining"


def json_safe(value: Any) -> Any:
    """Convert snapshot values to JSON-safe primitives without hiding missing data."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Series):
        return None
    if isinstance(value, (float, int)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    return value


@st.cache_data(ttl=900, show_spinner=False)
def fetch_us_equity(ticker: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, str | None]:
    """Fetch market data once, then calculate all visible analytics locally."""
    try:
        equity = yf.Ticker(ticker)
        history = equity.history(period="max", auto_adjust=False)
        benchmark_history = yf.Ticker(BENCHMARK).history(period="max", auto_adjust=False)
        if history.empty:
            return {}, history, benchmark_history, f"No market history found for {ticker}."
        if benchmark_history.empty:
            return {}, history, benchmark_history, "S&P 500 benchmark history is unavailable."

        info = equity.info
        last_close = float(history["Close"].iloc[-1])
        live_price = as_number(info.get("currentPrice")) or as_number(info.get("regularMarketPrice"))
        return {
            "current_price": live_price or last_close,
            "trailing_pe": as_number(info.get("trailingPE")),
            "price_to_book": as_number(info.get("priceToBook")),
            "free_cash_flow": as_number(info.get("freeCashflow")),
            "market_cap": as_number(info.get("marketCap")),
            "average_volume": as_number(info.get("averageVolume")),
            "return_on_equity": as_number(info.get("returnOnEquity")),
            "operating_margin": as_number(info.get("operatingMargins")),
            "revenue_growth": as_number(info.get("revenueGrowth")),
            "company_name": str(info.get("longName") or ticker),
            "sector": str(info.get("sector") or "No data"),
            "currency": str(info.get("currency") or "USD"),
            "exchange": str(info.get("exchange") or ""),
            "raw_fundamentals": {
                "freeCashflow": as_number(info.get("freeCashflow")),
                "returnOnEquity": as_number(info.get("returnOnEquity")),
                "operatingMargins": as_number(info.get("operatingMargins")),
                "revenueGrowth": as_number(info.get("revenueGrowth")),
            },
        }, history, benchmark_history, None
    except Exception as exc:
        return {}, pd.DataFrame(), pd.DataFrame(), str(exc)


def tradingview_exchange(yahoo_exchange: str) -> str:
    """Map Yahoo's common US exchange codes to TradingView display symbols."""
    return TRADINGVIEW_EXCHANGE_MAP.get(yahoo_exchange, "NASDAQ")


def show_tradingview_chart(symbol: str) -> None:
    """Render TradingView's official display-only widget with its attribution intact."""
    config = json.dumps(
        {
            "autosize": True,
            "symbol": symbol,
            "interval": "D",
            "timezone": "exchange",
            "theme": "light",
            "style": "1",
            "locale": "en",
            "withdateranges": True,
            "hide_side_toolbar": False,
            "allow_symbol_change": False,
            "save_image": False,
            "calendar": False,
            "details": True,
            "hotlist": False,
            "studies": ["MASimple@tv-basicstudies"],
            "support_host": "https://www.tradingview.com",
        }
    )
    chart_url = f"https://www.tradingview.com/chart/?symbol={quote(symbol, safe='')}"
    symbol_label = escape(symbol)
    widget_html = f"""
    <div class="tradingview-widget-container" style="height:540px;width:100%">
      <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
      <div class="tradingview-widget-copyright">
        <a href="{chart_url}" rel="noopener nofollow" target="_blank">{symbol_label} chart</a> by TradingView
      </div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {config}
      </script>
    </div>
    """
    components.html(widget_html, height=570, scrolling=False)


def show_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>📊 US Equity Research Desk</h1>
          <p>Transparent research controls for a disciplined investment process — S&P 500 benchmark · end-of-day market data</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_label(score: float | None) -> str:
    if score is None:
        return "Insufficient data"
    if score >= 75:
        return "Strong evidence set"
    if score >= 60:
        return "Adequate for review"
    return "More research required"


def research_page() -> None:
    show_hero()
    st.warning(
        "Research tool only — it does not provide investment advice or execute trades. Yahoo Finance data can be delayed, revised, or incomplete."
    )

    st.sidebar.header("US equity research")
    ticker = normalize_us_ticker(st.sidebar.text_input("US ticker", value="KHC", key="us_ticker"))
    if st.sidebar.button("↻ Refresh market data", use_container_width=True):
        fetch_us_equity.clear()
        st.rerun()
    st.sidebar.caption("Benchmark: S&P 500 (^GSPC) · cache: 15 minutes")

    if not ticker:
        st.info("Enter a US ticker to begin research.")
        return

    with st.spinner(f"Loading {ticker} and S&P 500 history..."):
        company, history, benchmark_history, error = fetch_us_equity(ticker)
    if error:
        st.error(f"Data quality check failed: {error}")
        return

    indicators = calculate_indicators(history, benchmark_history)
    research_score = calculate_research_score(indicators, company["raw_fundamentals"])
    score = research_score["score"]

    st.markdown('<p class="section-kicker">DATA QUALITY & MARKET REGIME</p>', unsafe_allow_html=True)
    st.subheader(f"{company['company_name']} ({ticker})")
    overview = st.columns(6)
    overview[0].metric("Last price", format_money(company["current_price"]))
    overview[1].metric("Market cap", format_money(company["market_cap"], 0))
    overview[2].metric("Regime", indicators["regime"])
    overview[3].metric("1Y volatility", format_percent(indicators["annual_volatility"]))
    overview[4].metric("1Y beta", format_multiple(indicators["beta_1y"]))
    overview[5].metric("Score", f"{score:.0f}/100" if score is not None else "No data")
    st.caption(
        f"Last available session: {indicators['last_date']} · Sector: {company['sector']} · "
        f"Score coverage: {research_score['coverage']:.0%} of observable criteria"
    )
    data_age = (utc_now().date() - datetime.fromisoformat(indicators["last_date"]).date()).days
    if data_age > 7:
        st.error("Market data is more than seven calendar days old. Do not rely on it until the feed is refreshed.")

    st.markdown('<p class="section-kicker">PRICE ACTION</p>', unsafe_allow_html=True)
    st.subheader("Trend, momentum, and relative performance")
    indicator_col, fundamental_col = st.columns(2)
    with indicator_col:
        st.dataframe(
            pd.DataFrame(
                [
                    ("EMA 20", format_money(indicators["ema_20"])),
                    ("EMA 50", format_money(indicators["ema_50"])),
                    ("EMA 200", format_money(indicators["ema_200"])),
                    ("RSI (14)", format_multiple(indicators["rsi_14"])),
                    ("ATR (14)", format_money(indicators["atr_14"])),
                    ("ATR as price %", format_percent(indicators["atr_percent"])),
                    ("63D vs S&P 500", format_percent(indicators["relative_return_63d"])),
                    ("1Y max drawdown", format_percent(indicators["max_drawdown_1y"])),
                ],
                columns=["Metric", "Reading"],
            ),
            hide_index=True,
            use_container_width=True,
        )
    with fundamental_col:
        st.dataframe(
            pd.DataFrame(
                [
                    ("Trailing P/E", format_multiple(company["trailing_pe"])),
                    ("Price / book", format_multiple(company["price_to_book"])),
                    ("Free cash flow", format_money(company["free_cash_flow"], 0)),
                    ("Return on equity", format_percent(company["return_on_equity"])),
                    ("Operating margin", format_percent(company["operating_margin"])),
                    ("Revenue growth", format_percent(company["revenue_growth"])),
                    ("Average daily volume", f"{company['average_volume']:,.0f}" if company["average_volume"] is not None else "No data"),
                ],
                columns=["Fundamental", "Reading"],
            ),
            hide_index=True,
            use_container_width=True,
        )

    chart_controls = st.columns(2)
    with chart_controls[0]:
        timeframe = st.selectbox("EMA timeframe", ["วัน (Daily)", "สัปดาห์ (Weekly)", "เดือน (Monthly)"], key=f"timeframe_{ticker}")
    with chart_controls[1]:
        periods = st.multiselect("EMA lines", [20, 50, 100, 200], default=[20, 50, 200], key=f"ema_{ticker}_{timeframe}")
    chart = prepare_ema_chart(history, timeframe, periods)
    if not chart.empty:
        st.line_chart(chart, x_label="Date", y_label="Adjusted price (USD)", use_container_width=True)

    st.markdown('<p class="section-kicker">MANUAL CHART & EVENT CHECK</p>', unsafe_allow_html=True)
    st.subheader("TradingView and Investing.com — display and research only")
    suggested_exchange = tradingview_exchange(company["exchange"])
    exchange_options = ["NASDAQ", "NYSE", "AMEX", "ARCA"]
    exchange = st.selectbox(
        "TradingView exchange",
        exchange_options,
        index=exchange_options.index(suggested_exchange) if suggested_exchange in exchange_options else 0,
        key=f"tv_exchange_{ticker}",
    )
    tv_symbol = f"{exchange}:{ticker}"
    resource_links = st.columns(3)
    resource_links[0].link_button(
        "Open full TradingView chart",
        f"https://www.tradingview.com/chart/?symbol={quote(tv_symbol, safe='')}",
        use_container_width=True,
    )
    resource_links[1].link_button(
        "Open Investing.com calendar",
        "https://www.investing.com/economic-calendar/",
        use_container_width=True,
    )
    resource_links[2].link_button(
        "Search ticker on Investing.com",
        f"https://www.investing.com/search/?q={quote(ticker, safe='')}",
        use_container_width=True,
    )
    show_tradingview = st.toggle(
        "Load TradingView chart in this page",
        value=False,
        help="This is a display-only visual check. Its values are never used by this app's score, risk engine, or backtest.",
        key=f"show_tradingview_{ticker}",
    )
    if show_tradingview:
        show_tradingview_chart(tv_symbol)
    st.caption(
        "TradingView and Investing.com are opened for manual review only. Do not treat displayed prices, alerts, or indicators as inputs to this model."
    )

    st.markdown('<p class="section-kicker">REPRODUCIBLE SCORECARD</p>', unsafe_allow_html=True)
    st.subheader(score_label(score))
    score_col, evidence_col = st.columns((0.65, 1.35))
    with score_col:
        st.metric("Observable evidence score", f"{score:.0f} / 100" if score is not None else "No data")
        st.metric("Data coverage", f"{research_score['coverage']:.0%}")
        st.caption("This is a checklist score, not a price target or buy/sell signal.")
    with evidence_col:
        st.dataframe(research_score["evidence"], hide_index=True, use_container_width=True)

    st.markdown('<p class="section-kicker">POSITION RISK ENGINE</p>', unsafe_allow_html=True)
    st.subheader("Size the risk before considering an order")
    plan_col, risk_col = st.columns((1.05, 0.95))
    default_entry = float(company["current_price"] or indicators["price"])
    default_stop = max(0.01, default_entry - 2 * (indicators["atr_14"] or default_entry * 0.05))
    with plan_col:
        account_value = st.number_input("Portfolio capital (USD)", min_value=0.0, value=100_000.0, step=1_000.0, key="portfolio_capital")
        risk_percent = st.number_input("Risk per trade (%)", min_value=0.05, max_value=5.0, value=0.50, step=0.05, key="risk_percent")
        max_position_percent = st.number_input("Maximum position (%)", min_value=1.0, max_value=100.0, value=10.0, step=1.0, key="max_position_percent")
        entry = st.number_input("Planned entry", min_value=0.01, value=round(default_entry, 2), step=0.01, key=f"entry_{ticker}")
        stop = st.number_input("Stop loss", min_value=0.01, value=round(default_stop, 2), step=0.01, key=f"stop_{ticker}")
        target = st.number_input("Target price", min_value=0.01, value=round(entry + 2 * (entry - stop), 2), step=0.01, key=f"target_{ticker}")

    risk_per_share = entry - stop
    reward_per_share = target - entry
    rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0
    position = calculate_position_risk(
        indicators["daily_returns"], account_value, risk_percent / 100, entry, stop, max_position_percent / 100
    )
    with risk_col:
        st.metric("Risk / reward", f"1 : {rr_ratio:.2f}")
        st.metric("Maximum shares", f"{position['shares']:,}")
        st.metric("Position value", format_money(float(position["position_value"])))
        st.metric("Loss at stop", format_money(float(position["shares"]) * max(risk_per_share, 0)))
        tail_1, tail_2 = st.columns(2)
        tail_1.metric("1-day historical VaR 95%", format_money(position["var_95"]))
        tail_2.metric("1-day historical CVaR 95%", format_money(position["cvar_95"]))
        st.caption("VaR/CVaR use the last 252 daily returns and historical outcomes; they are not maximum-loss guarantees.")
        if risk_percent > 1:
            st.warning("Risk per trade above 1%: confirm this fits your written risk policy.")
        if risk_per_share <= 0 or reward_per_share <= 0:
            st.error("A valid plan requires stop < entry < target.")
        elif rr_ratio < 2:
            st.warning("Risk/reward is below the 1:2 minimum configured for this process.")

    st.markdown('<p class="section-kicker">HISTORICAL STRATEGY REVIEW</p>', unsafe_allow_html=True)
    with st.expander("Backtest EMA 20/50 trend filter", expanded=False):
        backtest_col, cost_col = st.columns(2)
        with backtest_col:
            years = st.selectbox("Lookback", [3, 5, 10], index=1, format_func=lambda value: f"{value} years")
        with cost_col:
            cost_bps = st.number_input("Estimated one-way transaction cost (bps)", min_value=0.0, value=10.0, step=1.0)
        backtest_chart, backtest = run_ema_backtest(history, years, cost_bps)
        if backtest_chart.empty:
            st.info("At least one year of usable daily history is required for this backtest.")
        else:
            test_metrics = st.columns(5)
            test_metrics[0].metric("Strategy annualized return", format_percent(backtest["annual_return"]))
            test_metrics[1].metric("Buy & hold annualized", format_percent(backtest["benchmark_return"]))
            test_metrics[2].metric("Strategy volatility", format_percent(backtest["annual_volatility"]))
            test_metrics[3].metric("Max drawdown", format_percent(backtest["max_drawdown"]))
            test_metrics[4].metric("Trades", str(backtest["trades"]))
            st.line_chart(backtest_chart, x_label="Date", y_label="Growth of $1", use_container_width=True)
            st.caption("Signals are shifted by one trading day to avoid look-ahead bias. Results exclude taxes and use the transaction cost above; past results do not predict future results.")

    st.markdown('<p class="section-kicker">GOVERNANCE & AUDIT</p>', unsafe_allow_html=True)
    st.subheader("Human review before any action")
    source_col, event_col = st.columns(2)
    with source_col:
        news_source = st.text_input("Primary research source", value="SEC filing / company IR")
        is_verified = st.checkbox("I verified the thesis against primary-source evidence")
        chart_reviewed = st.checkbox("I reviewed the chart manually (TradingView or another charting tool)")
    with event_col:
        has_event = st.checkbox("Material event is pending (earnings, Fed, CPI, etc.)")
        calendar_reviewed = st.checkbox("I reviewed the economic calendar manually")
        cooling_minutes = st.selectbox("Cooling-off period", [30, 60], format_func=lambda value: f"{value} minutes")
        if st.button("Start cooling-off period"):
            st.session_state[cooling_key(ticker, cooling_minutes)] = utc_now()
            st.rerun()
        cooling_complete, cooling_message = cooling_status(ticker, cooling_minutes)
        st.info(f"Cooling-off: {cooling_message}")
    thesis = st.text_area("Thesis and invalidation condition", placeholder="What must be true? What evidence would invalidate the thesis?")
    no_emotion = st.checkbox("The proposed action is not driven by FOMO, panic, or a need to recover losses")

    ready_for_review = all((
        is_verified,
        not has_event,
        cooling_complete,
        bool(thesis.strip()),
        no_emotion,
        chart_reviewed,
        calendar_reviewed,
        risk_per_share > 0,
        reward_per_share > 0,
        rr_ratio >= 2,
        research_score["coverage"] >= 0.60,
    ))
    decision = "RESEARCH READY FOR HUMAN REVIEW" if ready_for_review else "HOLD / RESEARCH REQUIRED"
    if ready_for_review:
        st.success(f"{decision} — this remains a human decision, not an automated trade approval.")
    else:
        st.warning(f"{decision} — complete every applicable control before acting.")

    audit_snapshot = json.dumps(
        json_safe({
            "generated_at": utc_now().isoformat(),
            "benchmark": BENCHMARK,
            "indicators": indicators,
            "research_score": {"score": score, "coverage": research_score["coverage"]},
            "position_risk": position,
            "manual_checks": {
                "chart_reviewed": chart_reviewed,
                "calendar_reviewed": calendar_reviewed,
                "primary_source_verified": is_verified,
            },
            "backtest": backtest if "backtest" in locals() else None,
        }),
        ensure_ascii=False,
    )
    if st.button("Save research record", type="primary", use_container_width=True):
        record_id = save_decision(
            ticker=ticker,
            news_headline=thesis,
            news_source=news_source,
            has_event=has_event,
            is_verified=is_verified,
            current_price=company["current_price"],
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            risk_reward=rr_ratio,
            cooling_done=cooling_complete,
            thesis_clear=bool(thesis.strip()),
            no_emotion=no_emotion,
            decision=decision,
            notes=(
                f"Score: {format_multiple(score)} | Coverage: {research_score['coverage']:.0%} | "
                f"Chart reviewed: {chart_reviewed} | Calendar reviewed: {calendar_reviewed}"
            ),
            analysis_snapshot=audit_snapshot,
        )
        st.success(f"Research record #{record_id} saved locally.")


def journal_page() -> None:
    show_hero()
    st.subheader("Research audit trail")
    records = get_decisions()
    if not records:
        st.info("No records yet. Save a research record from the US equity research page.")
        return
    table = pd.DataFrame(records)
    visible = [
        "created_at", "ticker", "decision", "current_price", "entry_price", "stop_loss",
        "target_price", "risk_reward", "is_verified", "cooling_done", "notes",
    ]
    st.dataframe(table[visible], hide_index=True, use_container_width=True)
    st.download_button(
        "Download audit trail (CSV)",
        table.to_csv(index=False).encode("utf-8-sig"),
        "us-equity-research-audit.csv",
        "text/csv",
    )


def methodology_page() -> None:
    show_hero()
    st.subheader("Model boundaries and methodology")
    st.markdown(
        """
- **Data:** end-of-day Yahoo Finance market data with S&P 500 as the benchmark. It is not a licensed institutional real-time feed.
- **Risk:** volatility, beta, drawdown, ATR, historical VaR/CVaR, position caps, and stop-loss risk are calculated from available history.
- **Signal:** EMA, RSI, scorecards, and backtests are reproducible screens—not recommendations or forecasts.
- **Governance:** every saved record retains the inputs, decision controls, and a JSON snapshot of calculated readings.
- **Next institutional step:** connect licensed data, add portfolio holdings/correlation limits, and require authenticated review workflows.
        """
    )


initialize_database()
st.sidebar.title("US Research Desk")
page = st.sidebar.radio("Navigation", ["Research", "Audit trail", "Methodology"], label_visibility="collapsed")
if page == "Research":
    research_page()
elif page == "Audit trail":
    journal_page()
else:
    methodology_page()
