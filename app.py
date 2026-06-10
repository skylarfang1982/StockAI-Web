import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 網頁基本設定
st.set_page_config(page_title="多模型決策系統", layout="wide")
st.title("🔮 多模型 AI 綜合決策系統")
st.write("本系統同時運算【趨勢、動能、波動度】三大獨立量化模型，進行交叉驗證投票。")

# 網頁格子輸入元件
ticker_input = st.text_input("請輸入股票代號（台股如 2330.TW，美股如 NVDA）", value="2330.TW")

if st.button("啟動多模型交叉分析"):
    with st.spinner('三大模型全力運算中...'):
        # 1. 抓取數據 (近兩年數據)
        df = yf.download(ticker_input, start="2024-01-01")
        
        if df.empty:
            st.error("無法抓取該股票數據，請檢查代號是否正確。")
        else:
            # 清洗欄位，避免新版 yfinance 多層索引問題
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

            # ----------------------------------------------------
            # 【模型一：趨勢+籌碼流】 (MACD + OBV + MA20)
            # ----------------------------------------------------
            df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD_Hist'] = df['EMA12'] - df['EMA26'] - (df['EMA12'] - df['EMA26']).ewm(span=9, adjust=False).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            price_diff = df['Close'].diff()
            direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
            df['OBV'] = (direction * df['Volume']).cumsum()
            df['OBV_MA5'] = df['OBV'].rolling(window=5).mean()

            # ----------------------------------------------------
            # 【模型二：動能反轉流】 (RSI + KD 指標)
            # ----------------------------------------------------
            # RSI 計算
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            # KD 計算
            low_min = df['Low'].rolling(window=9).min()
            high_max = df['High'].rolling(window=9).max()
            df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
            df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
            df['D'] = df['K'].ewm(com=2, adjust=False).mean()

            # ----------------------------------------------------
            # 【模型三：波動度突破流】 (布林通道 Bollinger Bands)
            # ----------------------------------------------------
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
            df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)

            # ----------------------------------------------------
            # 🎯 決策投票核心大腦 (計算最新一天的訊號)
            # ----------------------------------------------------
            # 模型 1 判斷
            m1_buy = (df['MACD_Hist'].iloc[-1] > 0) and (df['OBV'].iloc[-1] > df['OBV_MA5'].iloc[-1]) and (df['Close'].iloc[-1] > df['MA20'].iloc[-1])
            m1_sell = (df['MACD_Hist'].iloc[-1] < 0) or (df['Close'].iloc[-1] < df['MA20'].iloc[-1])
            m1_score = 1 if m1_buy else (-1 if m1_sell else 0)

            # 模型 2 判斷 (低檔超賣反彈)
            m2_buy = (df['RSI'].iloc[-1] < 40) and (df['K'].iloc[-1] > df['D'].iloc[-1])
            m2_sell = (df['RSI'].iloc[-1] > 70) and (df['K'].iloc[-1] < df['D'].iloc[-1])
            m2_score = 1 if m2_buy else (-1 if m2_sell else 0)

            # 模型 3 判斷 (強勢突破上軌)
            m3_buy = df['Close'].iloc[-1] >= df['BB_Upper'].iloc[-1]
            m3_sell = df['Close'].iloc[-1] <= df['BB_Lower'].iloc[-1]
            m3_score = 1 if m3_buy else (-1 if m3_sell else 0)

            # 綜合勝率計分板
            total_votes = [m1_score, m2_score, m3_score]
            buy_votes = total_votes.count(1)
            sell_votes = total_votes.count(-1)
            
            # 計算看漲勝率機率
            confidence_score = int((buy_votes / 3) * 100)

            # ----------------------------------------------------
            # 🌐 前端網頁視覺化呈現
            # ----------------------------------------------------
            latest_price = round(df['Close'].iloc[-1], 1)
            latest_date = df.index[-1].strftime('%Y-%m-%d')

            st.subheader(f"📊 綜合診斷報告 (分析日期: {latest_date})")
            
            # 三大指標看板
            col1, col2, col3 = st.columns(3)
            col1.metric("當前收盤價", f"{latest_price} 元")
            col2.metric("綜合看漲信心度", f"{confidence_score} %")
            
            if confidence_score >= 66:
                col3.success("🔥 強烈建議買入 (共識度高)")
            elif confidence_score == 33:
                col3.info("🍏 偏多續抱 / 少量試單")
            elif sell_votes >= 2:
                col3.error("🚨 建議避險 / 減碼賣出")
            else:
                col3.warning("💤 趨勢不明 / 觀望盤整")

            st.divider()

            # 模型分項解析明細
            st.write("### 🔍 各模型獨立審查明細")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.write("**1. 趨勢籌碼流 (MACD+OBV)**")
                if m1_score == 1:
                    st.write("🟢 趨勢看多")
                else:
                    st.write("🔴 趨勢看空/轉弱")
                
            with c2:
                st.write("**2. 動能反轉流 (RSI+KD)**")
                if m2_score == 1:
                    st.write("🟢 處於低檔反彈區")
                elif m2_score == -1:
                    st.write("🔴 處於高檔過熱區")
                else:
                    st.write("🟡 訊號中立")
                
            with c3:
                st.write("**3. 波動突破流 (布林通道)**")
                if m3_score == 1:
                    st.write("🟢 帶量突破上軌 (強勢股)")
                elif m3_score == -1:
                    st.write("🔴 跌破下軌 (弱勢格局)")
                else:
                    st.write("🟡 通道內震盪")

            st.divider()

            # 🛠 摺疊元件：預設不顯示圖表，點擊才展開
            with st.expander("📊 點擊查看歷史 K 線與波動度通道走勢圖 (布林通道)"):
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(df['Close'], label='收盤價', color='black', alpha=0.7)
                ax.plot(df['BB_Upper'], label='布林上軌 (壓力線)', color='red', linestyle='--', alpha=0.5)
                ax.plot(df['BB_Middle'], label='布林中軌 (月線)', color='orange', alpha=0.5)
                ax.plot(df['BB_Lower'], label='布林下軌 (支撐線)', color='green', linestyle='--', alpha=0.5)
                ax.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.05)
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.2)
                st.pyplot(fig)
