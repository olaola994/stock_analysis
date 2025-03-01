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
    st.sidebar.header("Investing strategy")
    strategy = st.sidebar.selectbox("Select strategy:", ["RSI", "SMA", "Bollinger Bands"])



    data = yf.download(symbol, start=start_date, end=end_date)

    # Moving Averages
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()

    # Bollinger Bands
    data["BB_Middle"] = data["Close"].rolling(window=20, min_periods=1).mean()
    rolling_std = data["Close"].rolling(window=20, min_periods=1).std().squeeze()
    data["BB_Upper"] = data["BB_Middle"] + 2 * rolling_std
    data["BB_Lower"] = data["BB_Middle"] - 2 * rolling_std

    # RSI
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

    # MACD
    # data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    # data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    # data["MACD"] = data["EMA_12"] - data["EMA_26"]
    # data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

    def AI_decision(row):
        def safe_get(row, column):
            value = row.at[column] if column in row else None
            if isinstance(value, pd.Series):
                value = value.iloc[0] 
            return float(value) if pd.notna(value) else None

        rsi = safe_get(row, "RSI")
        sma_10 = safe_get(row, "SMA_10")
        sma_50 = safe_get(row, "SMA_50")
        macd = safe_get(row, "MACD")
        signal_line = safe_get(row, "Signal_Line")
        close = safe_get(row, "Close")
        bb_lower = safe_get(row, "BB_Lower")
        bb_upper = safe_get(row, "BB_Upper")
        
        if strategy == "RSI":
            if rsi is not None and rsi < 30:
                return "BUY"
            elif rsi is not None and rsi > 70:
                return "SELL"
        elif strategy == "SMA":
            if sma_10 is not None and sma_50 is not None:
                if sma_10 > sma_50:
                    return "BUY"
                elif sma_10 < sma_50:
                    return "SELL"
        # elif strategy == "MACD":
        #     if macd is not None and signal_line is not None:
        #         if macd > signal_line:
        #             return "BUY"
        #         elif macd < signal_line:
        #             return "SELL"
        elif strategy == "Bollinger Bands":
            if close is not None and bb_lower is not None and bb_upper is not None:
                if close < bb_lower:
                    return "BUY"
                elif close > bb_upper:
                    return "SELL"
        return "HOLD"
    
    data["Decision"] = data.apply(AI_decision, axis=1)

    st.write("📜 Stock Data:")
    st.dataframe(data.iloc[:, :5]) 

    st.download_button(
    label="📥 Download stock data as CSV",
    data=data.to_csv().encode("utf-8"),
    file_name=f"{symbol}_stock_data.csv",
    mime="text/csv"
    )

    buy_signals = data[data["Decision"] == "BUY"]
    sell_signals = data[data["Decision"] == "SELL"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(data.index, data["Close"], label="Close price", color="blue")
    ax1.plot(data.index, data["EMA_10"], linestyle="--", label="EMA 10", color="black")
    ax1.plot(data.index, data["EMA_50"], linestyle="--", label="EMA 50", color="purple")

    ax1.scatter(buy_signals.index, buy_signals["Close"], color="green", marker="^", label="BUY", alpha=1, s=100)
    ax1.scatter(sell_signals.index, sell_signals["Close"], color="red", marker="v", label="SELL", alpha=1, s=100)

    ax1.set_title(f"Close price {symbol}")
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

    if strategy == "RSI":
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data["Close"], label="Close price", color="blue")
    ax.plot(data.index, data["SMA_10"], linestyle="--", label="SMA 10", color="black")
    ax.plot(data.index, data["SMA_50"], linestyle="--", label="SMA 50", color="purple")
    ax.scatter(buy_signals.index, buy_signals["Close"], color="green", marker="^", label="BUY", alpha=1, s=100)
    ax.scatter(sell_signals.index, sell_signals["Close"], color="red", marker="v", label="SELL", alpha=1, s=100)
    ax.set_title(f"Close price {symbol}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()

    if strategy == "SMA":
        st.pyplot(fig)


    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(data.index, data["Close"], label="Close price", color="blue")
    ax.plot(data.index, data["BB_Upper"], linestyle="--", label="BB_Upper", color="black")
    ax.plot(data.index, data["BB_Lower"], linestyle="--", label="BB_Lower", color="purple")
    ax.scatter(buy_signals.index, buy_signals["Close"], color="green", marker="^", label="BUY", alpha=1, s=100)
    ax.scatter(sell_signals.index, sell_signals["Close"], color="red", marker="v", label="SELL", alpha=1, s=100)
    ax.set_title(f"Close {symbol}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()

    if strategy == "Bollinger Bands":
        st.pyplot(fig)


    df = data.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]
    df = df.dropna()

    model = Prophet(
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        daily_seasonality=False
    )

    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.add_seasonality(name="weekly", period=7, fourier_order=3)
    model.add_country_holidays(country_name='US')

    model.fit(df)

    future = model.make_future_dataframe(periods=30, freq="B")

    forecast = model.predict(future)
    fig_forecast, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["ds"], df["y"], label="Historical prices", color="blue", marker="o")
    ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="green", linestyle="--")
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="lightgreen", alpha=0.3, label="Confidence interval (95%)")

    ax.set_title(f"Share price forecast {symbol} for the next 30 days", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Closing price ($)", fontsize=12)
    ax.legend()

    st.pyplot(fig_forecast)


    
else:
    st.warning("🔴 Wybierz **pełny zakres dat** ")