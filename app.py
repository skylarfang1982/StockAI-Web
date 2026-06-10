import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 網頁基本設定
st.set_page_config(page_title="量化決策系統", layout="centered")
st.title("📊 AI量化波段決策系統")
st.write("輸入股票代號，自動分析技術面與籌碼面，給出買賣決策建議。")

# 1. 網頁格子輸入元件 (預設台積電)
ticker_input = st.text_input("請輸入股票代號（台股請加 .TW，美股直接輸入代號）", value="2330.TW")

if st.button("開始分析"):
    with st.spinner('正在獲取最新市場數據並計算中...'):
        # 抓取數據
        df = yf.download(ticker_input, start="2024-01-01")
        
        if df.empty:
            st.error("無法抓取該股票數據，請檢查代號是否正確（例如：2317.TW 或 AAPL）。")
        else:
            # 清洗欄位
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

            # 技術面：MACD 與 20MA
            df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['DIF'] = df['EMA12'] - df['EMA26']
            df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['DIF'] - df['MACD_Signal'] 
            df['MA20'] = df['Close'].rolling(window=20).mean()

            # 籌碼面：OBV
            price_diff = df['Close'].diff()
            direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
            df['OBV'] = (direction * df['Volume']).cumsum()
            df['OBV_MA5'] = df['OBV'].rolling(window=5).mean()

            # 邏輯大腦
            df['Signal'] = 0
            for i in range(1, len(df)):
                macd_cross_up = (df['MACD_Hist'].iloc[i-1] < 0) and (df['MACD_Hist'].iloc[i] > 0)
                volume_inflow = df['OBV'].iloc[i] > df['OBV_MA5'].iloc[i]
                above_ma20 = df['Close'].iloc[i] > df['MA20'].iloc[i]
                
                macd_cross_down = (df['MACD_Hist'].iloc[i-1] > 0) and (df['MACD_Hist'].iloc[i] < 0)
                below_ma20 = df['Close'].iloc[i] < df['MA20'].iloc[i]

                if macd_cross_up and volume_inflow and above_ma20:
                    df.loc[df.index[i], 'Signal'] = 1
                elif macd_cross_down or below_ma20:
                    df.loc[df.index[i], 'Signal'] = -1

            df['Position'] = df['Signal'].replace(0, np.nan).ffill().shift(1).fillna(0)
            
            # 當日狀態分析
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            latest_price = round(df['Close'].iloc[-1], 1)
            current_signal = df['Signal'].iloc[-1]
            current_pos = df['Position'].iloc[-1]

            # 網頁視覺化儀表板看板 (KPI Metrics)
            col1, col2 = st.columns(2)
            col1.metric("最新收盤價", f"{latest_price} 元", f"日期: {latest_date}")
            
            if current_signal == 1:
                col2.success("💡 建議買入")
                st.info("原因：今日觸發黃金交叉，且資金集聚、站穩月線多頭格局。")
            elif current_signal == -1:
                col2.error("🚨 建議賣出")
                st.warning("原因：趨勢轉弱或股價跌破 20MA 防守線，建議停損/落袋為安。")
            else:
                if current_pos == 1:
                    col2.info("🍏 持股續抱")
                else:
                    col2.warning("💤 空方觀望")

            # 繪製線圖並顯示於網頁
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['Close'], label='收盤價', color='black', alpha=0.4)
            ax.plot(df['MA20'], label='20MA波段線', color='orange', linestyle='--')
            
            buy_signals = df[df['Signal'] == 1]
            ax.scatter(buy_signals.index, buy_signals['Close'], label='買入訊號', marker='^', color='green', s=100)
            sell_signals = df[df['Signal'] == -1]
            ax.scatter(sell_signals.index, sell_signals['Close'], label='賣出訊號', marker='v', color='red', s=100)
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
