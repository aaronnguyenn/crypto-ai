# app.py
"""
Streamlit Crypto Price Tracker
---------------------------------
Run with:
    streamlit run app.py

Requirements:
    pip install streamlit requests pandas
"""

import streamlit as st
import requests
import pandas as pd

# --------------------------------- CONFIGURATION ---------------------------------
st.set_page_config(page_title="Crypto Price Tracker", page_icon="💰", layout="centered")

# --------------------------------- CONSTANTS ---------------------------------
SUPPORTED_COINS = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Binance Coin (BNB)": "binancecoin",
    "Solana (SOL)": "solana",
    "Cardano (ADA)": "cardano",
    "XRP": "ripple",
}
DEFAULT_COINS = ["Bitcoin (BTC)", "Ethereum (ETH)"]

# --------------------------------- HELPERS ---------------------------------
@st.cache_data(ttl=300)
def fetch_current_prices(ids: list[str], vs_currency: str) -> dict:
    """Fetch current crypto prices from CoinGecko.
    Cached for 5 minutes (300 seconds).
    """
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": ",".join(ids), "vs_currencies": vs_currency}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()

@st.cache_data(ttl=300)
def fetch_historical_prices(coin_id: str, vs_currency: str, days: int) -> pd.DataFrame:
    """Return a DataFrame of historical prices for *coin_id* over *days*.
    Index is timestamp, column is price.
    Cached for 5 minutes.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    prices = response.json().get("prices", [])

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

# --------------------------------- UI ---------------------------------
st.title("💰 Live Crypto Price Tracker")

vs_currency = st.selectbox("View prices in:", ["usd", "eur", "vnd"], index=0, format_func=str.upper)

selected_coins = st.multiselect(
    "Choose cryptocurrencies:",
    list(SUPPORTED_COINS.keys()),
    default=DEFAULT_COINS,
)

days = st.slider("Historical range (days)", min_value=1, max_value=365, value=30)

if selected_coins:
    coin_ids = [SUPPORTED_COINS[c] for c in selected_coins]

    # ---------- Current Prices ----------
    try:
        current_data = fetch_current_prices(coin_ids, vs_currency)
    except Exception as e:
        st.error(f"Error fetching current prices: {e}")
        st.stop()

    cols = st.columns(len(selected_coins))
    for i, coin_label in enumerate(selected_coins):
        cid = SUPPORTED_COINS[coin_label]
        price = current_data.get(cid, {}).get(vs_currency)
        cols[i].metric(label=coin_label, value=f"{price:,} {vs_currency.upper()}")

    # ---------- Historical Chart ----------
    chart_coin_label = st.selectbox("Select a coin for historical chart", selected_coins)
    chart_coin_id = SUPPORTED_COINS[chart_coin_label]

    try:
        hist_df = fetch_historical_prices(chart_coin_id, vs_currency, days)
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        st.stop()

    st.subheader(f"{chart_coin_label} price over last {days} day(s)")
    st.line_chart(hist_df["price"], height=350)

else:
    st.info("Select at least one cryptocurrency to view prices.")

st.caption("Data source: CoinGecko API · Updates every 5 minutes")
