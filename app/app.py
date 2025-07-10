"""
Streamlit app: Rate-Rise Strategy Evaluator
Author: Chaordichris & ChatGPT (o3)
Updated: 2025‑07‑10



Run with:  streamlit run streamlit_rates_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Rates Strategy Evaluator", layout="wide")

###############################################################################
# Utility helpers
###############################################################################
CACHE_DAYS = 3  # re-download prices if older than this many days

@st.cache_data(show_spinner=False, ttl=CACHE_DAYS * 24 * 60 * 60)
def get_price_history(ticker: str, days_back: int = 365):
    """Download *adjusted* price history and return as Series.
    Falls back to raw Close if Adj Close is missing (futures & many indices)."""
    end = date.today()
    start = end - timedelta(days=days_back)

    # auto_adjust=True gives adjusted Close for equities; harmless for futures
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        st.warning(f"No data pulled for {ticker} – check symbol or yfinance availability.")
        return pd.Series(dtype=float)

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    return df[price_col].rename(ticker)

###############################################################################
# Strategy classes – each must expose: description, inputs(), compute(), results_df
###############################################################################
class ShortFuture:
    name = "Short Treasury Future"

    CONTRACTS = {
        "TY (10‑Y)": {"ticker": "ZN=F", "dv01": 85},
        "FV (5‑Y)":  {"ticker": "ZF=F", "dv01": 55},
        "UB (Ultra‑Bond)": {"ticker": "UB=F", "dv01": 160},
    }

    def inputs(self):
        self.contract = st.selectbox("Contract", list(self.CONTRACTS))
        self.lots = st.number_input("Number of contracts", 1, 500, 1, step=1)
        self.scn_bps = st.slider("Scenario – parallel yield shift (bp)", 0, 200, 25, step=5)

    def compute(self):
        info = self.CONTRACTS[self.contract]
        self.hist = get_price_history(info["ticker"])
        if self.hist.empty:
            self.last_px = np.nan
            self.dv01 = self.pnl = 0
            return
        self.last_px = float(self.hist.iloc[-1])
        self.dv01 = info["dv01"] * self.lots
        self.pnl = self.dv01 * self.scn_bps

    def results_df(self):
        return pd.DataFrame({
            "Strategy": [self.name],
            "Contract": [self.contract],
            "Lots": [self.lots],
            "DV01 ($/bp)": [self.dv01],
            "Scenario Δy (bp)": [self.scn_bps],
            "PnL ($)": [self.pnl],
        })

    def chart(self):
        if self.hist.empty:
            st.info("No price data to chart.")
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.hist.index, y=self.hist.values, name=self.contract))
        fig.update_layout(title=f"{self.contract} price history", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

class TltPut:
    name = "TLT Put Option"

    def inputs(self):
        self.expiry_year = st.selectbox("Expiry", [2026, 2027, 2028])
        self.strike = st.number_input("Strike ($)", 60, 120, 85)
        self.contracts = st.number_input("# option contracts", 1, 500, 10)
        self.premium = st.number_input("Premium per contract ($)", 0.1, 50.0, 8.5, step=0.1,
                                        help="Fill with actual mid‑market premium from broker.")
        self.scn_price = st.slider("Scenario – TLT price at expiry ($)", 50, 120, 80)

    def compute(self):
        self.hist = get_price_history("TLT")
        if self.hist.empty:
            self.last_px = np.nan
            self.payoff = 0
            return
        self.last_px = float(self.hist.iloc[-1])
        intrinsic = max(0, self.strike - self.scn_price)
        self.payoff = (intrinsic - self.premium) * self.contracts * 100  # 100 shares/option

    def results_df(self):
        return pd.DataFrame({
            "Strategy": [self.name],
            "Strike": [self.strike],
            "Contracts": [self.contracts],
            "Premium/ct": [self.premium],
            "Scenario TLT": [self.scn_price],
            "Payoff ($)": [self.payoff],
        })

    def chart(self):
        if self.hist.empty:
            st.info("No price data to chart.")
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.hist.index, y=self.hist.values, name="TLT"))
        fig.update_layout(title="TLT price history", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

class KalshiBinary:
    name = "Kalshi Binary"

    def inputs(self):
        self.question = st.text_input("Contract description", "Fed upper bound ≥ 5.00 % on Dec 2025 FOMC")
        self.market_price = st.number_input("Market price (¢)", 1.0, 99.0, 38.0, step=0.5)
        self.notional = st.number_input("Contracts size ($1 payout)", 10, 10000, 100)

    def compute(self):
        cost = self.market_price / 100 * self.notional
        payout = self.notional  # if event occurs
        self.pnl = payout - cost

    def results_df(self):
        return pd.DataFrame({
            "Strategy": [self.name],
            "Contract": [self.question],
            "Nos ($1)": [self.notional],
            "Mkt price (¢)": [self.market_price],
            "Max Profit ($)": [self.pnl],
            "Max Loss ($)": [-(self.market_price / 100 * self.notional)],
        })

    def chart(self):
        st.markdown("Binary contracts have no historical price chart – PnL is all‑or‑nothing.")

###############################################################################
# Strategy registry and router
###############################################################################
registry = {cls.name: cls for cls in (ShortFuture, TltPut, KalshiBinary)}

st.title("📈 Rate‑Rise Strategy Evaluator")
st.markdown("Interactively size trades, pull live prices, and stress‑test P/L under yield scenarios. **Educational use only.**")

with st.sidebar:
    st.header("Select a strategy")
    choice = st.selectbox("Strategy type", list(registry))
    st.divider()

strategy = registry[choice]()
strategy.inputs()

if st.sidebar.button("Compute / Refresh", type="primary"):
    strategy.compute()
    res = strategy.results_df()

    st.subheader("📑 Results")
    st.dataframe(res, use_container_width=True, hide_index=True, height=220)

    st.subheader("📊 Underlying Price History")
    strategy.chart()

    st.caption("DV01 values are approximations. Option pricing uses intrinsic – replace with full Black‑Scholes/76 if desired.")

with st.expander("ℹ️ Disclaimers and Next Steps"):
    st.markdown("""
    • This tool illustrates mechanics only and is **not** investment advice.
    • Prices pulled from free APIs (yfinance) may lag; confirm with your broker.
    • Extend by adding: payer swaptions, curve spreads, or portfolio aggregation.
    • Replace the naïve option payoff with Black‑76

