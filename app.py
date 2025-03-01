import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import datetime
from prophet import Prophet

st.title("Stock analysis")
symbol = st.text_input("Specify the stock exchange symboly:", "AAPL")

today = datetime.datetime.now().date()
prev_year = today.replace(year=today.year - 1)
    
duration = st.date_input(
    "Choose date range",
    (prev_year, today), 
    min_value=datetime.date(2000, 1, 1),
    max_value=today
)
if isinstance(duration, tuple) and len(duration) == 2:
    start_date, end_date = duration
    st.write(f"📅 **Date Range:** {start_date} - {end_date}")
    st.sidebar.header("Choose indicators")
    show_macd = st.sidebar.checkbox("Show MACD")


    data = yf.download(symbol, start=start_date, end=end_date)
    data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["Delta"] = data["Close"].diff()
    data["U"] = data["Delta"].apply(lambda x: x if x > 0 else 0)
    data["D"] = data["Delta"].apply(lambda x: -x if x < 0 else 0)
    data["U_SUM"] = data["U"].rolling(window=14).sum()
    data["D_SUM"] = data["D"].rolling(window=14).sum()
    data["RS"] = data["U_SUM"]/data["D_SUM"]
    data["RS"] = data["RS"].replace([float('inf'), -float('inf')], 100)
    data["RSI"] = 100 - (100 / (1 + data["RS"]))
    def AI_decision(rsi):
        if rsi < 30:
            return "BUY"
        elif rsi > 70:
            return "SELL"
        else:
            return "HOLD"
    data["Decision"] = data["RSI"].apply(AI_decision)

    st.write("📜 Stock Data:")
    st.dataframe(data)
    st.download_button(
    label="📥 Download stock data as CSV",
    data=data.to_csv().encode("utf-8"),
    file_name=f"{symbol}_stock_data.csv",
    mime="text/csv"
    )

    buy_signals = data[data["Decision"] == "BUY"]
    sell_signals = data[data["Decision"] == "SELL"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(data.index, data["Close"], label="Closing price", color="blue")
    ax1.plot(data.index, data["EMA_10"], label="EMA 10", color="green")
    ax1.plot(data.index, data["EMA_50"], label="EMA 50", color="red")

    ax1.scatter(buy_signals.index, buy_signals["Close"], color="green", marker="^", label="BUY", alpha=1, s=100)
    ax1.scatter(sell_signals.index, sell_signals["Close"], color="red", marker="v", label="SELL", alpha=1, s=100)

    ax1.set_title(f"Closing price {symbol}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price ($)")
    ax1.legend()

    ax2.plot(data.index, data["RSI"], label="RSI", color="black")
    ax2.axhline(70, linestyle="--", color="red", label="Overbought (70)")
    ax2.axhline(30, linestyle="--", color="green", label="Oversold (30)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("RSI value")
    ax2.legend()
    ax2.set_title(f"RSI")


    st.pyplot(fig)


    df = data.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]

    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig_forecast, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["ds"], df["y"], label="Historical prices", color="blue", marker="o")
    ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="green", linestyle="--")
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="lightgreen", alpha=0.3, label="Confidence interval (95%)")

    ax.set_title(f"📈 Share price forecast {symbol} for the next 30 days", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Closing price ($)", fontsize=12)
    ax.legend()

    st.pyplot(fig_forecast)

    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

    fig_macd, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data["MACD"], label="MACD", color="blue")
    ax.plot(data.index, data["Signal_Line"], label="Signal Line", color="red")
    ax.axhline(0, linestyle="--", color="black")
    ax.set_title(f"📈 MACD Indicator for {symbol}")
    ax.set_xlabel("Date")
    ax.set_ylabel("MACD Value")
    ax.legend()
    

    if show_macd:
        st.pyplot(fig_macd)

    
else:
    st.warning("🔴 Wybierz **pełny zakres dat** ")