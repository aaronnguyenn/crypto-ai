import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Thiết lập trang
st.set_page_config(
    page_title="Ứng dụng Phân tích Cryptocurrency",
    page_icon="📈",
    layout="wide"
)

# Tiêu đề ứng dụng
st.title("📊 Ứng dụng Phân tích Cryptocurrency")

# Danh sách các cryptocurrency phổ biến
cryptocurrencies = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Binance Coin": "BNB-USD",
    "Cardano": "ADA-USD",
    "Solana": "SOL-USD",
    "XRP": "XRP-USD",
    "Polkadot": "DOT-USD", 
    "Dogecoin": "DOGE-USD",
    "Avalanche": "AVAX-USD",
    "Chainlink": "LINK-USD"
}

# Sidebar cho cài đặt
st.sidebar.header("Tùy chọn hiển thị")

# Lựa chọn cryptocurrency
selected_crypto = st.sidebar.selectbox("Chọn Cryptocurrency", list(cryptocurrencies.keys()))

# Lựa chọn khoảng thời gian
time_periods = {
    "7 ngày": 7,
    "1 tháng": 30,
    "3 tháng": 90,
    "6 tháng": 180,
    "1 năm": 365,
    "5 năm": 1825
}
selected_period = st.sidebar.selectbox("Chọn khoảng thời gian", list(time_periods.keys()))

# Tùy chọn hiển thị
show_volume = st.sidebar.checkbox("Hiển thị khối lượng giao dịch", value=True)
show_indicators = st.sidebar.checkbox("Hiển thị chỉ báo kỹ thuật", value=True)

# Hiển thị thông tin về cryptocurrency đã chọn
st.header(f"{selected_crypto} ({cryptocurrencies[selected_crypto]})")

# Tải dữ liệu
@st.cache_data(ttl=300)  # Cache trong 5 phút
def load_data(ticker, period_days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

with st.spinner('Đang tải dữ liệu...'):
    data = load_data(cryptocurrencies[selected_crypto], time_periods[selected_period])

if data.empty:
    st.error("Không thể tải dữ liệu. Vui lòng thử lại sau.")
else:
    # Hiển thị thông tin hiện tại
    col1, col2, col3, col4 = st.columns(4)
    
    last_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = last_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    col1.metric("Giá hiện tại", f"${last_price:.2f}", f"{price_change_pct:.2f}%")
    col2.metric("Giá cao nhất (24h)", f"${data['High'].iloc[-1]:.2f}")
    col3.metric("Giá thấp nhất (24h)", f"${data['Low'].iloc[-1]:.2f}")
    col4.metric("Khối lượng (24h)", f"{data['Volume'].iloc[-1]:,.0f}")

    # Biểu đồ giá
    st.subheader("Biểu đồ giá")
    
    # Tạo biểu đồ candlestick
    fig = go.Figure()
    
    # Thêm candlesticks
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Candlestick"
    ))
    
    # Thêm đường trung bình động nếu được chọn
    if show_indicators:
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['MA20'], 
            mode='lines', 
            name='MA20',
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['MA50'], 
            mode='lines', 
            name='MA50',
            line=dict(color='blue', width=1)
        ))
    
    # Cập nhật layout
    fig.update_layout(
        title=f"{selected_crypto} Chart - {selected_period}",
        xaxis_title="Ngày",
        yaxis_title="Giá (USD)",
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Biểu đồ khối lượng giao dịch
    if show_volume:
        st.subheader("Khối lượng giao dịch")
        volume_fig = px.bar(
            data,
            x=data.index,
            y='Volume',
            color_discrete_sequence=['rgba(0, 255, 0, 0.5)']
        )
        volume_fig.update_layout(
            title=f"{selected_crypto} - Khối lượng giao dịch",
            xaxis_title="Ngày",
            yaxis_title="Khối lượng",
            height=300,
            template="plotly_dark"
        )
        st.plotly_chart(volume_fig, use_container_width=True)
    
    # Phân tích kỹ thuật
    if show_indicators:
        st.subheader("Phân tích kỹ thuật")
        
        # Tính RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Tính MACD
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['Signal']
        
        # Hiển thị các chỉ báo kỹ thuật
        col1, col2 = st.columns(2)
        
        # RSI Chart
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['RSI'], 
            mode='lines', 
            name='RSI',
            line=dict(color='purple', width=1)
        ))
        
        # Thêm đường tham chiếu
        rsi_fig.add_shape(
            type="line", line=dict(dash='dash', color="red"),
            y0=70, y1=70, x0=data.index[0], x1=data.index[-1]
        )
        rsi_fig.add_shape(
            type="line", line=dict(dash='dash', color="green"),
            y0=30, y1=30, x0=data.index[0], x1=data.index[-1]
        )
        
        rsi_fig.update_layout(
            title=f"RSI (14)",
            xaxis_title="Ngày",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),
            height=300,
            template="plotly_dark"
        )
        
        col1.plotly_chart(rsi_fig, use_container_width=True)
        
        # MACD Chart
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['MACD'], 
            mode='lines', 
            name='MACD',
            line=dict(color='blue', width=1)
        ))
        
        macd_fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['Signal'], 
            mode='lines', 
            name='Signal',
            line=dict(color='orange', width=1)
        ))
        
        # Thêm histogram
        colors = ['green' if val >= 0 else 'red' for val in data['MACD_Hist']]
        macd_fig.add_trace(go.Bar(
            x=data.index,
            y=data['MACD_Hist'],
            name='Histogram',
            marker_color=colors
        ))
        
        macd_fig.update_layout(
            title=f"MACD (12,26,9)",
            xaxis_title="Ngày",
            yaxis_title="MACD",
            height=300,
            template="plotly_dark"
        )
        
        col2.plotly_chart(macd_fig, use_container_width=True)
    
    # Hiển thị dữ liệu thô
    with st.expander("Xem dữ liệu thô"):
        st.dataframe(data)
        
        # Tải xuống dữ liệu dưới dạng CSV
        csv = data.to_csv().encode('utf-8')
        st.download_button(
            label="Tải xuống dữ liệu CSV",
            data=csv,
            file_name=f'{selected_crypto}_data.csv',
            mime='text/csv',
        )

    # So sánh với các crypto khác
    st.subheader("So sánh hiệu suất")
    comparison_cryptos = st.multiselect(
        "Chọn cryptocurrencies để so sánh", 
        list(cryptocurrencies.keys()),
        default=[list(cryptocurrencies.keys())[0], list(cryptocurrencies.keys())[1]] if len(cryptocurrencies) > 1 else [list(cryptocurrencies.keys())[0]]
    )
    
    if comparison_cryptos:
        comparison_fig = go.Figure()
        
        for crypto in comparison_cryptos:
            comp_data = load_data(cryptocurrencies[crypto], time_periods[selected_period])
            # Chuẩn hóa giá để so sánh hiệu suất (100 = giá ngày đầu tiên)
            normalized_data = comp_data['Close'] / comp_data['Close'].iloc[0] * 100
            
            comparison_fig.add_trace(go.Scatter(
                x=comp_data.index,
                y=normalized_data,
                mode='lines',
                name=crypto
            ))
        
        comparison_fig.update_layout(
            title="So sánh hiệu suất (chuẩn hóa)",
            xaxis_title="Ngày",
            yaxis_title="Hiệu suất (%)",
            height=500,
            template="plotly_dark"
        )
        
        st.plotly_chart(comparison_fig, use_container_width=True)

# Thêm thông tin tác giả
st.sidebar.markdown("---")
st.sidebar.info("""
**Về ứng dụng này**
Ứng dụng này giúp phân tích giá và xu hướng của các cryptocurrency phổ biến.
""")
st.sidebar.warning("Lưu ý: Thông tin được cung cấp chỉ mang tính chất tham khảo và không phải là lời khuyên đầu tư.")
