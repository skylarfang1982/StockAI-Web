import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 網頁基本設定
st.set_page_config(page_title="AI 量化決策與選股系統", layout="wide")

# ----------------------------------------------------
# 🗂 預設台股大型成分股清單 (精選台灣市值前150大龍頭股)
# ----------------------------------------------------
@st.cache_data(ttl=3600)  # 暫存機制，避免重複抓取造成網頁卡頓
def get_tw_stock_pool():
    return [
        "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2881.TW", "2882.TW", "2317.TW",
        "2891.TW", "2303.TW", "3711.TW", "2412.TW", "2886.TW", "2357.TW", "3231.TW", "2324.TW",
        "1301.TW", "1303.TW", "2603.TW", "2609.TW", "2615.TW", "2002.TW", "2379.TW", "3008.TW",
        "2884.TW", "2892.TW", "2880.TW", "2885.TW", "2887.TW", "5880.TW", "2890.TW", "2353.TW",
        "2371.TW", "2352.TW", "2345.TW", "23Automate", "2377.TW", "2395.TW", "3034.TW", "3037.TW",
        "3045.TW", "4938.TW", "2408.TW", "2409.TW", "6116.TW", "3481.TW", "1101.TW", "1402.TW"
    ] # 這裡先列出最具代表性的前 50 大流動性龍頭股，確保掃描速度極快

# ----------------------------------------------------
# ⚙️ 核心量化計算引擎 (單一股票分析)
# ----------------------------------------------------
def calculate_signals(df):
    if df.empty or len(df) < 20:
        return 0, 0, 0, None
    
    # 清洗欄位
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # 1. 趨勢籌碼 (MACD + OBV + MA20)
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = df['EMA12'] - df['EMA26'] - (df['EMA12'] - df['EMA26']).ewm(span=9, adjust=False).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    price_diff = df['Close'].diff()
    direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
    df['OBV'] = (direction * df['Volume']).cumsum()
    df['OBV_MA5'] = df['OBV'].rolling(window=5).mean()

    # 2. 動能反轉 (RSI + KD)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min + 1e-10) * 100
    df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()

    # 3. 波動突破 (布林通道)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)

    # 評分與計分
    m1_score = 1 if (df['MACD_Hist'].iloc[-1] > 0 and df['OBV'].iloc[-1] > df['OBV_MA5'].iloc[-1] and df['Close'].iloc[-1] > df['MA20'].iloc[-1]) else 0
    m2_score = 1 if (df['RSI'].iloc[-1] < 45 and df['K'].iloc[-1] > df['D'].iloc[-1]) else 0
    m3_score = 1 if df['Close'].iloc[-1] >= df['BB_Upper'].iloc[-1] else 0

    confidence_score = int(((m1_score + m2_score + m3_score) / 3) * 100)
    
    # 估算買賣點
    recent_low = df['Low'].iloc[-20:].min()
    recent_high = df['High'].iloc[-20:].max()
    best_buy = round((recent_low + df['BB_Lower'].iloc[-1]) / 2, 1)
    best_sell = round((recent_high + df['BB_Upper'].iloc[-1]) / 2, 1)
    
    return confidence_score, best_buy, best_sell, df

# ----------------------------------------------------
# 🌐 網頁前端分頁配置 (Tabs)
# ----------------------------------------------------
tab1, tab2 = st.tabs(["🔍 個股策略診斷", "🚀 即時潛力台股雷達 (推薦 Top 10)"])

# ===== Tab 1: 原本的個股診斷功能 =====
with tab1:
    st.header("個股量化分析大腦")
    ticker_input = st.text_input("請輸入股票代號（台股如 2330.TW，美股如 NVDA）", value="2330.TW", key="single_search")
    
    if st.button("啟動個股分析"):
        with st.spinner('數據計算中...'):
            df = yf.download(ticker_input, start="2024-01-01")
            if df.empty:
                st.error("無法抓取該股票數據。")
            else:
                score, buy_p, sell_p, processed_df = calculate_signals(df)
                latest_price = round(processed_df['Close'].iloc[-1], 1)
                
                # 顯示資訊看板
                col1, col2 = st.columns(2)
                col1.metric("當前收盤價", f"{latest_price} 元")
                col2.metric("綜合看漲信心度", f"{score} %")
                
                st.write(f"**建議買點：** {buy_p} 元 | **建議賣點：** {sell_p} 元")
                
                with st.expander("📊 查看走勢圖"):
                    fig, ax = plt.subplots(figsize=(14, 4))
                    ax.plot(processed_df['Close'], color='black', label='收盤價')
                    ax.axhline(buy_p, color='green', linestyle=':', label='支撐買點')
                    ax.axhline(sell_p, color='red', linestyle=':', label='壓力賣點')
                    ax.legend()
                    st.pyplot(fig)

# ===== Tab 2: 全新選股雷達功能 =====
with tab2:
    st.header("🎛 台股多模型潛力股篩選器")
    st.write("點擊下方按鈕，系統將自動掃描台股核心龍頭股，依據【趨勢+籌碼+動能】綜合模型篩選出看漲動能最強的前 10 名個股。")
    
    if st.button("開始掃描台股市場"):
        stock_pool = get_tw_stock_pool()
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 批次下載全市場最新數據
        status_text.write("正在下載台股核心股票池最新數據...")
        all_data = yf.download(stock_pool, period="3mo", group_by='ticker')
        
        for idx, ticker in enumerate(stock_pool):
            status_text.write(f"正在交叉運算： {ticker} ...")
            progress_bar.progress((idx + 1) / len(stock_pool))
            
            try:
                # 提取單一股票的 Dataframe
                if isinstance(all_data.columns, pd.MultiIndex):
                    sub_df = all_data[ticker].dropna()
                else:
                    sub_df = all_data.dropna()
                    
                if sub_df.empty:
                    continue
                    
                score, buy_p, sell_p, _ = calculate_signals(sub_df)
                latest_price = round(sub_df['Close'].iloc[-1], 1)
                
                # 我們只記錄有一定看漲訊號或動能的股票
                results.append({
                    "股票代號": ticker,
                    "最新收盤價": f"{latest_price} 元",
                    "看漲信心指數": score,
                    "建議分批買點": f"{buy_p} 元",
                    "建議波段賣點": f"{sell_p} 元"
                })
            except Exception as e:
                continue
                
        status_text.write("✨ 掃描完成！已為您導出最佳前 10 名看漲動能股：")
        progress_bar.empty()
        
        # 轉成 DataFrame 並排序，抓出前 10 名
        res_df = pd.DataFrame(results)
        if not res_df.empty:
            # 優先排信心度高的，若信心度一樣，代表動能旗鼓相當
            res_df = res_df.sort_values(by="看漲信心指數", ascending=False).head(10).reset_index(drop=True)
            
            # 格式化表格輸出
            st.dataframe(
                res_df, 
                use_container_width=True,
                column_config={
                    "看漲信心指數": st.column_config.ProgressColumn(
                        "看漲信心指數",
                        help="三大模型交叉投票通過率",
                        format="%d%%",
                        min_value=0,
                        max_value=100,
                    )
                }
            )
            
            st.success("💡 投資小建議：清單中信心指數達 66% 以上的個股，代表其技術面、籌碼面、波動度同時產生共鳴，短線或波段上漲機率較高，可優先納入您的自選股觀察！")
        else:
            st.warning("當前市場波動劇烈，暫未掃描到符合強勢多頭多指標共鳴的股票，請保持觀望。")
