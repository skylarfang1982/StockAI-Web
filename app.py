import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 網頁基本設定
st.set_page_config(page_title="多模型決策系統", layout="wide")
st.title("🔮 多模型 AI 綜合決策系統")
st.write("本系統同時運算【趨勢、動能、波動度】三大獨立量化模型，並動態估算當前最佳動態買賣點。")

# 網頁格子輸入元件
ticker_input = st.text_input("請輸入股票代號（台股如 2330.TW，美股如 NVDA）", value="2330.TW")

if st.button("啟動多模型交叉分析"):
    with st.spinner('數據計算與最佳價位估算中...'):
        # 1. 抓取數據
        df = yf.download(ticker_input, start="2024-01-01")
        
        if df.empty:
            st.error("無法抓取該股票數據，請檢查代號是否正確。")
        else:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

            # ----------------------------------------------------
            # 【模型指標運算】
            # ----------------------------------------------------
            # 趨勢籌碼
            df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD_Hist'] = df['EMA12'] - df['EMA26'] - (df['EMA12'] - df['EMA26']).ewm(span=9, adjust=False).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            price_diff = df['Close'].diff()
            direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
            df['OBV'] = (direction * df['Volume']).cumsum()
            df['OBV_MA5'] = df['OBV'].rolling(window=5).mean()

            # 動能反轉
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            low_min = df['Low'].rolling(window=9).min()
            high_max = df['High'].rolling(window=9).max()
            df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
            df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
            df['D'] = df['K'].ewm(com=2, adjust=False).mean()

            # 波動突破
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
            df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)

            # ----------------------------------------------------
            # 🎯 決策投票核心大腦
            # ----------------------------------------------------
            m1_buy = (df['MACD_Hist'].iloc[-1] > 0) and (df['OBV'].iloc[-1] > df['OBV_MA5'].iloc[-1]) and (df['Close'].iloc[-1] > df['MA20'].iloc[-1])
            m1_sell = (df['MACD_Hist'].iloc[-1] < 0) or (df['Close'].iloc[-1] < df['MA20'].iloc[-1])
            m1_score = 1 if m1_buy else (-1 if m1_sell else 0)

            m2_buy = (df['RSI'].iloc[-1] < 40) and (df['K'].iloc[-1] > df['D'].iloc[-1])
            m2_sell = (df['RSI'].iloc[-1] > 70) and (df['K'].iloc[-1] < df['D'].iloc[-1])
            m2_score = 1 if m2_buy else (-1 if m2_sell else 0)

            m3_buy = df['Close'].iloc[-1] >= df['BB_Upper'].iloc[-1]
            m3_sell = df['Close'].iloc[-1] <= df['BB_Lower'].iloc[-1]
            m3_score = 1 if m3_buy else (-1 if m3_sell else 0)

            total_votes = [m1_score, m2_score, m3_score]
            buy_votes = total_votes.count(1)
            sell_votes = total_votes.count(-1)
            confidence_score = int((buy_votes / 3) * 100)

            # ----------------------------------------------------
            # 💰 【新增核心】動態最佳買賣點估算
            # ----------------------------------------------------
            latest_price = round(df['Close'].iloc[-1], 1)
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            
            # 最佳買點：結合布林下軌與近20日低點（支撐區）
            recent_low = df['Low'].iloc[-20:].min()
            bb_lower_now = df['BB_Lower'].iloc[-1]
            best_buy_price = round((recent_low + bb_lower_now) / 2, 1)
            
            # 最佳賣點：結合布林上軌與近20日高點（壓力區）
            recent_high = df['High'].iloc[-20:].max()
            bb_upper_now = df['BB_Upper'].iloc[-1]
            best_sell_price = round((recent_high + bb_upper_now) / 2, 1)

            # ----------------------------------------------------
            # 🌐 前端網頁視覺化呈現
            # ----------------------------------------------------
            st.subheader(f"📊 綜合診斷報告 (分析日期: {latest_date})")
            
            # 第一層：主指標看板
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

            # 第二層：新增的「最佳買賣價位估算」區塊
            st.write("### 🎯 現階段最佳動態策略價位估算")
            buy_col, sell_col, action_col = st.columns(3)
            
            with buy_col:
                st.markdown(f"##### 🟢 最佳分批買點 (強支撐位)")
                st.markdown(f"## **{best_buy_price} 元**")
                st.caption("估算邏輯：近20日波段低點與布林通道下軌之核心支撐帶。")
                
            with sell_col:
                st.markdown(f"##### 🔴 最佳波段賣點 (強壓力位)")
                st.markdown(f"## **{best_sell_price} 元**")
                st.caption("估算邏輯：近20日波段高點與布林通道上軌之密集賣壓區。")
                
            with action_col:
                st.markdown("##### ✏️ 策略空間操作指引")
                if latest_price <= best_buy_price * 1.02:
                    st.success("✨ 目前股價已極度接近「最佳買點」，適合左側逢低佈局！")
                elif latest_price >= best_sell_price * 0.98:
                    st.error("⚠️ 目前股價已高度逼近「最佳賣點」，請勿追高，宜分批獲利入袋！")
                else:
                    # 計算距離買賣點的空間比率
                    pct_to_buy = round(((latest_price - best_buy_price) / latest_price) * 100, 1)
                    st.info(f"當前價格處於合理震盪區間。距離下方最佳買點約有 **{pct_to_buy}%** 的修正空間。")

            st.divider()

            # 第三層：模型分項解析明細
            st.write("### 🔍 各模型獨立審查明細")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**1. 趨勢籌碼流 (MACD+OBV)**")
                st.write("🟢 趨勢看多" if m1_score == 1 else "🔴 趨勢看空/轉弱")
            with c2:
                st.write("**2. 動能反轉流 (RSI+KD)**")
                st.write("🟢 處於低檔反彈區" if m2_score == 1 else ("🔴 處於高檔過熱區" if m2_score == -1 else "🟡 訊號中立"))
            with c3:
                st.write("**3. 波動突破流 (布林通道)**")
                st.write("🟢 帶量突破上軌 (強勢股)" if m3_score == 1 else ("🔴 跌破下軌 (弱勢格局)" if m3_score == -1 else "🟡 通道內震盪"))

            st.divider()

            # 第四層：摺疊元件（預設隱藏圖表）
            with st.expander("📊 點擊查看歷史 K 線與波動度通道走勢圖 (布林通道)"):
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(df['Close'], label='收盤價', color='black', alpha=0.7)
                ax.plot(df['BB_Upper'], label='布林上軌 (壓力線 - 參考賣點)', color='red', linestyle='--', alpha=0.5)
                ax.plot(df['BB_Middle'], label='布林中軌 (月線)', color='orange', alpha=0.5)
                ax.plot(df['BB_Lower'], label='布林下軌 (支撐線 - 參考買點)', color='green', linestyle='--', alpha=0.5)
                ax.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.05)
                
                # 在最新的日期標註估算的最佳買賣點橫線
                ax.axhline(best_buy_price, color='green', linestyle=':', alpha=0.7, label=f'即時支撐買點: {best_buy_price}')
                ax.axhline(best_sell_price, color='red', linestyle=':', alpha=0.7, label=f'即時壓力賣點: {best_sell_price}')
                
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.2)
                st.pyplot(fig)
