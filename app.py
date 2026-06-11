import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 網頁基本設定
st.set_page_config(page_title="AI 量化決策與選股系統", layout="wide")

# 利用 CSS 縮小手機端標題字體，並優化 UI 緊湊度
st.markdown("""
    <style>
    .mobile-title {
        font-size: 24px !important;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .stMetric label {
        font-size: 14px !important;
    }
    .stMetric div {
        font-size: 20px !important;
    }
    div[data-testid="stExpander"] div p {
        font-size: 13px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="mobile-title">🔮 多模型 AI 綜合決策系統</p>', unsafe_allow_html=True)

# ----------------------------------------------------
# 🗂 150檔核心股池（升級版：包含中文名稱與板塊對應）
# ----------------------------------------------------
def get_stock_pool_details():
    # 完美對應字典：代號 -> {"名稱": 中文名稱, "板塊": 產業分類}
    mapping = {
        # --- 半導體與 IC 設計 ---
        "2330.TW": {"名稱": "台積電", "板塊": "半導體科技"},
        "2454.TW": {"名稱": "聯發科", "板塊": "半導體科技"},
        "2303.TW": {"名稱": "聯電", "板塊": "半導體科技"},
        "3711.TW": {"名稱": "日月光投控", "板塊": "半導體科技"},
        "2379.TW": {"名稱": "瑞昱", "板塊": "半導體科技"},
        "3034.TW": {"名稱": "聯詠", "板塊": "半導體科技"},
        "2408.TW": {"名稱": "南亞科", "板塊": "半導體科技"},
        "2337.TW": {"名稱": "旺宏", "板塊": "半導體科技"},
        "2449.TW": {"名稱": "京元電子", "板塊": "半導體科技"},
        "3532.TW": {"名稱": "台勝科", "板塊": "半導體科技"},
        "3035.TW": {"名稱": "智原", "板塊": "半導體科技"},
        "5269.TW": {"名稱": "祥碩", "板塊": "半導體科技"},
        "6415.TW": {"名稱": "矽力*-KY", "板塊": "半導體科技"},
        "5347.TW": {"名稱": "世界", "板塊": "半導體科技"},
        "6147.TW": {"名稱": "淇譽電", "板塊": "半導體科技"},
        "8299.TW": {"名稱": "群聯", "板塊": "半導體科技"},
        
        # --- AI伺服器、電腦與電子代工 ---
        "2317.TW": {"名稱": "鴻海", "板塊": "AI電腦與代工"},
        "2382.TW": {"名稱": "廣達", "板塊": "AI電腦與代工"},
        "2357.TW": {"名稱": "華碩", "板塊": "AI電腦與代工"},
        "3231.TW": {"名稱": "緯創", "板塊": "AI電腦與代工"},
        "2324.TW": {"名稱": "仁寶", "板塊": "AI電腦與代工"},
        "2353.TW": {"名稱": "宏碁", "板塊": "AI電腦與代工"},
        "2371.TW": {"名稱": "大同", "板塊": "AI電腦與代工"},
        "2352.TW": {"名稱": "佳世達", "板塊": "AI電腦與代工"},
        "2345.TW": {"名稱": "智邦", "板塊": "AI電腦與代工"},
        "2377.TW": {"名稱": "微星", "板塊": "AI電腦與代工"},
        "2395.TW": {"名稱": "研華", "板塊": "AI電腦與代工"},
        "4938.TW": {"名稱": "和碩", "板塊": "AI電腦與代工"},
        "6669.TW": {"名稱": "緯穎", "板塊": "AI電腦與代工"},
        "2354.TW": {"名稱": "鴻準", "板塊": "AI電腦與代工"},
        "2355.TW": {"名稱": "敬鵬", "板塊": "AI電腦與代工"},
        "2376.TW": {"名稱": "技嘉", "板塊": "AI電腦與代工"},
        "3017.TW": {"名稱": "奇鋐", "板塊": "AI電腦與代工"},
        "3264.TW": {"名稱": "欣銓", "板塊": "AI電腦與代工"},
        "3596.TW": {"名稱": "智易", "板塊": "AI電腦與代工"},
        
        # --- 建材營造板塊 ---
        "2542.TW": {"名稱": "興富發", "板塊": "建材營造傳統"},
        "2548.TW": {"名稱": "華固", "板塊": "建材營造傳統"},
        "9945.TW": {"名稱": "潤泰新", "板塊": "建材營造傳統"},
        "2501.TW": {"名稱": "國建", "板塊": "建材營造傳統"},
        "2534.TW": {"名稱": "宏盛", "板塊": "建材營造傳統"},
        
        # --- 電機機械與重電綠能 ---
        "1504.TW": {"名稱": "東元", "板塊": "電機重電綠能"},
        "1513.TW": {"名稱": "中興電", "板塊": "電機重電綠能"},
        "1519.TW": {"名稱": "華城", "板塊": "電機重電綠能"},
        "1605.TW": {"名稱": "華新", "板塊": "電機重電綠能"},
        "9958.TW": {"名稱": "世紀鋼", "板塊": "電機重電綠能"},
        
        # --- 航運、航空與物流 ---
        "2603.TW": {"名稱": "長榮", "板塊": "航運航空物流"},
        "2609.TW": {"名稱": "陽明", "板塊": "航運航空物流"},
        "2615.TW": {"名稱": "萬海", "板塊": "航運航空物流"},
        "2618.TW": {"名稱": "長榮航", "板塊": "航運航空物流"},
        "2610.TW": {"名稱": "華航", "板塊": "航運航空物流"},
        "2633.TW": {"名稱": "高鐵", "板塊": "航運航空物流"},
        "2605.TW": {"名稱": "新興", "板塊": "航運航空物流"},
        "2606.TW": {"名稱": "裕民", "板塊": "航運航空物流"},
        
        # --- 金控與銀行 ---
        "2881.TW": {"名稱": "富邦金", "板塊": "金融保險金控"},
        "2882.TW": {"名稱": "國泰金", "板塊": "金融保險金控"},
        "2891.TW": {"名稱": "中國信託", "板塊": "金融保險金控"},
        "2886.TW": {"名稱": "兆豐金", "板塊": "金融保險金控"},
        "2884.TW": {"名稱": "玉山金", "板塊": "金融保險金控"},
        "2892.TW": {"名稱": "第一金", "板塊": "金融保險金控"},
        "2880.TW": {"名稱": "華南金", "板塊": "金融保險金控"},
        "2885.TW": {"名稱": "元大金", "板塊": "金融保險金控"},
        "2887.TW": {"名稱": "台新金", "板塊": "金融保險金控"},
        "5880.TW": {"名稱": "合庫金", "板塊": "金融保險金控"},
        "2890.TW": {"名稱": "永豐金", "板塊": "金融保險金控"},
        "5876.TW": {"名稱": "上海商銀", "板塊": "金融保險金控"},
        "2801.TW": {"名稱": "彰銀", "板塊": "金融保險金控"},
        "2809.TW": {"名稱": "京城銀", "板塊": "金融保險金控"},
        "2812.TW": {"名稱": "台中銀", "板塊": "金融保險金控"},
        "2834.TW": {"名稱": "臺企銀", "板塊": "金融保險金控"},
        "2838.TW": {"名稱": "聯邦銀", "板塊": "金融保險金控"},
        "2845.TW": {"名稱": "遠東銀", "板塊": "金融保險金控"},
        "2855.TW": {"名稱": "統一證", "板塊": "金融保險金控"},
        "2883.TW": {"名稱": "開發金", "板塊": "金融保險金控"},
        "2888.TW": {"名稱": "新光金", "板塊": "金融保險金控"},
        "2889.TW": {"名稱": "國票金", "板塊": "金融保險金控"},
        
        # --- 電子零組件、ABF、PCB與通路 ---
        "2308.TW": {"名稱": "台達電", "板塊": "電子零組件通路"},
        "3037.TW": {"名稱": "欣興", "板塊": "電子零組件通路"},
        "2327.TW": {"名稱": "國巨", "板塊": "電子零組件通路"},
        "2360.TW": {"名稱": "致茂", "板塊": "電子零組件通路"},
        "2474.TW": {"名稱": "可成", "板塊": "電子零組件通路"},
        "3702.TW": {"名稱": "大聯大", "板塊": "電子零組件通路"},
        "2383.TW": {"名稱": "台光電", "板塊": "電子零組件通路"},
        "2385.TW": {"名稱": "群光", "板塊": "電子零組件通路"},
        "3044.TW": {"名稱": "健鼎", "板塊": "電子零組件通路"},
        "6213.TW": {"名稱": "聯茂", "板塊": "電子零組件通路"},
        "6269.TW": {"名稱": "台郡", "板塊": "電子零組件通路"},
        "3036.TW": {"名稱": "文曄", "板塊": "電子零組件通路"},
        "4958.TW": {"名稱": "臻鼎-KY", "板塊": "電子零組件通路"},
        "8046.TW": {"名稱": "南電", "板塊": "電子零組件通路"},
        
        # --- 生技醫療與化學 ---
        "1795.TW": {"名稱": "美時", "板塊": "生技醫療化學"},
        "4147.TW": {"名稱": "中裕", "板塊": "生技醫療化學"},
        "1707.TW": {"名稱": "葡萄王", "板塊": "生技醫療化學"},
        "1722.TW": {"名稱": "台肥", "板塊": "生技醫療化學"},
        "1717.TW": {"名稱": "長興", "板塊": "生技醫療化學"},
        
        # --- 光電與面板 ---
        "2409.TW": {"名稱": "面板友達", "板塊": "光電面板面板"},
        "3481.TW": {"名稱": "群創", "板塊": "光電面板面板"},
        "3008.TW": {"名稱": "大立光", "板塊": "光電面板面板"},
        "6116.TW": {"名稱": "彩晶", "板塊": "光電面板面板"},
        "3406.TW": {"名稱": "玉晶光", "板塊": "光電面板面板"},
        
        # --- 傳統工業（水泥、鋼鐵、塑化、基礎建設） ---
        "1101.TW": {"名稱": "台泥", "板塊": "傳統基礎工業"},
        "1402.TW": {"名稱": "遠東新", "板塊": "傳統基礎工業"},
        "2002.TW": {"名稱": "中鋼", "板塊": "傳統基礎工業"},
        "1301.TW": {"名稱": "台塑", "板塊": "傳統基礎工業"},
        "1303.TW": {"名稱": "南亞", "板塊": "傳統基礎工業"},
        "2105.TW": {"名稱": "正新", "板塊": "傳統基礎工業"},
        "2106.TW": {"名稱": "建大", "板塊": "傳統基礎工業"},
        "9933.TW": {"名稱": "中鼎", "板塊": "傳統基礎工業"},
        
        # --- 觀光、百貨、運動與內需消費 ---
        "2207.TW": {"名稱": "和泰車", "板塊": "內需消費生活"},
        "9910.TW": {"名稱": "豐泰", "板塊": "內需消費生活"},
        "2912.TW": {"名稱": "統一超", "板塊": "內需消費生活"},
        "5904.TW": {"名稱": "寶雅", "板塊": "內需消費生活"},
        "2707.TW": {"名稱": "晶華", "板塊": "內需消費生活"},
        "9921.TW": {"名稱": "巨大", "板塊": "內需消費生活"},
        "9914.TW": {"名稱": "美利達", "板塊": "內需消費生活"},
        "9904.TW": {"名稱": "寶成", "板塊": "內需消費生活"},
        "2201.TW": {"名稱": "裕隆", "板塊": "內需消費生活"},
        "2204.TW": {"名稱": "中華車", "板塊": "內需消費生活"}
    }
    return mapping

# ----------------------------------------------------
# ⚙️ 核心量化計算引擎
# ----------------------------------------------------
def calculate_signals(df):
    if df.empty or len(df) < 20:
        return 0, 0, 0, 0, 0, 0, None
    
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

    # 狀態計分
    m1_buy = (df['MACD_Hist'].iloc[-1] > 0 and df['OBV'].iloc[-1] > df['OBV_MA5'].iloc[-1] and df['Close'].iloc[-1] > df['MA20'].iloc[-1])
    m1_sell = (df['MACD_Hist'].iloc[-1] < 0 or df['Close'].iloc[-1] < df['MA20'].iloc[-1])
    m1_score = 1 if m1_buy else (-1 if m1_sell else 0)

    m2_buy = (df['K'].iloc[-1] > df['D'].iloc[-1] and df['RSI'].iloc[-1] > 45)
    m2_sell = (df['K'].iloc[-1] < df['D'].iloc[-1] and df['RSI'].iloc[-1] > 70)
    m2_score = 1 if m2_buy else (-1 if m2_sell else 0)

    m3_buy = df['Close'].iloc[-1] >= df['BB_Upper'].iloc[-1]
    m3_sell = df['Close'].iloc[-1] <= df['BB_Lower'].iloc[-1]
    m3_score = 1 if m3_buy else (-1 if m3_sell else 0)

    positive_signals = [1 for s in [m1_score, m2_score, m3_score] if s == 1]
    confidence_score = int((len(positive_signals) / 3) * 100)
    
    # 決策價位優化
    latest_close = df['Close'].iloc[-1]
    recent_low = df['Low'].iloc[-20:].min()
    recent_high = df['High'].iloc[-20:].max()
    
    best_buy = round(max(df['MA20'].iloc[-1], recent_low), 1)
    best_sell = round(max(latest_close * 1.08, recent_high, df['BB_Upper'].iloc[-1]), 1)
    
    if best_buy >= latest_close:
        best_buy = round(latest_close * 0.93, 1)
    if best_sell <= latest_close:
        best_sell = round(latest_close * 1.10, 1)
    
    return confidence_score, best_buy, best_sell, m1_score, m2_score, m3_score, df

# ----------------------------------------------------
# 🌐 網頁前端分頁配置
# ----------------------------------------------------
tab1, tab2 = st.tabs(["🔍 個股策略診斷", "🚀 全台股整體產業金流與強勢股雷達"])

# ===== Tab 1: 個股診斷功能 =====
with tab1:
    st.markdown("<p style='font-size:13px; color:gray;'>輸入特定股票代號，查看量化指標明細與動態買賣操作指引。</p>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ 查看後台量化決策模型架構", expanded=False):
        st.markdown("""
        本系統採用**三模組多重共鳴投票算法**：
        1. **趨勢與量能模型：** MACD柱狀體翻紅 + OBV超越5日均線 + 股價高於MA20。符合得1票。
        2. **動能與反轉模型：** KD黃金交叉 且 RSI > 45。符合得1票。
        3. **波動與突破模型：** 股價強勢帶量突破布林上軌。符合得1票。
        """)

    ticker_input = st.text_input("請輸入股票代號（如：2330.TW）", value="2330.TW")
    run_analysis = st.button("啟動個股分析")
    
    if "show_chart" not in st.session_state:
        st.session_state.show_chart = False
    if "current_ticker" not in st.session_state:
        st.session_state.current_ticker = ""

    if ticker_input != st.session_state.current_ticker:
        st.session_state.show_chart = False
        st.session_state.current_ticker = ticker_input

    if run_analysis:
        if not ticker_input:
            st.warning("請先輸入股票代號。")
        else:
            with st.spinner('數據計算中...'):
                df = yf.download(ticker_input, start="2024-01-01")
                if df.empty:
                    st.error("無法抓取該股票數據。")
                else:
                    score, buy_p, sell_p, m1, m2, m3, processed_df = calculate_signals(df)
                    
                    # 嘗試從核心池抓取中文名，若無則顯示代號
                    pool = get_stock_pool_details()
                    stock_name = pool[ticker_input]["名稱"] if ticker_input in pool else ticker_input
                    
                    st.session_state.analysis_data = {
                        "name": stock_name,
                        "score": score, "buy_p": buy_p, "sell_p": sell_p,
                        "latest_price": round(processed_df['Close'].iloc[-1], 1),
                        "latest_date": processed_df.index[-1].strftime('%Y-%m-%d'),
                        "chart_df": processed_df.tail(65)
                    }
                    st.session_state.show_chart = False

    if "analysis_data" in st.session_state:
        data = st.session_state.analysis_data
        
        st.markdown(f"### 📊 綜合診斷報告 - {data['name']} ({data['latest_date']})")
        col1, col2, col3 = st.columns(3)
        col1.metric("當前收盤價", f"{data['latest_price']} 元")
        col2.metric("看漲信心度", f"{data['score']}%")
        
        with col3:
            if data['score'] >= 66:
                st.success("🔥 建議買入")
            elif data['score'] == 33:
                st.info("🍏 偏多續抱")
            else:
                st.warning("💤 趨勢不明")

        st.divider()
        st.markdown("### 🎯 最佳動態策略價位")
        buy_col, sell_col = st.columns(2)
        buy_col.metric("🟢 最佳分批買點", f"{data['buy_p']} 元")
        sell_col.metric("🔴 最佳波段賣點", f"{data['sell_p']} 元")
        
        if data['latest_price'] <= data['buy_p'] * 1.02:
            st.success("✨ 股價接近買點，適合低佈！")
        elif data['latest_price'] >= data['sell_p'] * 0.98:
            st.error("⚠️ 股價逼近賣點，請勿追高！")

        st.divider()
        
        if st.session_state.show_chart:
            if st.button("👁️ 隱藏技術對照圖表"):
                st.session_state.show_chart = False
                st.rerun()
        else:
            if st.button("📊 顯示量化買賣對照圖表"):
                st.session_state.show_chart = True
                st.rerun()

        if st.session_state.show_chart:
            chart_df = data["chart_df"]
            buy_p = data["buy_p"]
            sell_p = data["sell_p"]
            
            fig, ax = plt.subplots(figsize=(7, 4))
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            ax.plot(chart_df.index, chart_df['Close'], label='收盤價', color='#1f77b4', linewidth=2)
            ax.plot(chart_df.index, chart_df['MA20'], label='MA20', color='#ff7f0e', linestyle='--')
            
            ax.axhline(y=buy_p, color='#2ca02c', linestyle=':', linewidth=1.5, label=f'買點:{buy_p}')
            ax.axhline(y=sell_p, color='#d62728', linestyle=':', linewidth=1.5, label=f'賣點:{sell_p}')
            
            ax.set_title(f"{ticker_input} ({data['name']}) 技術趨勢對照", fontsize=11, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            st.pyplot(fig)

# ===== Tab 2: 選股雷達功能 =====
with tab2:
    st.markdown("<p style='font-size:15px; font-weight:bold;'>🚀 產業金流與強勢股雷達</p>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ 查看 150 檔大數據混合計算邏輯", expanded=False):
        st.markdown("""
        1. **整體產業板塊資金強度：** 該板塊內所有個股信心指數的平均值。
        2. **強勢個股 Top 20：** 撈取多模組投票獲得高分的前 20 檔強勢個股。
        """)
        
    if st.button("開始全面掃描 150 檔核心市場"):
        stock_pool_details = get_stock_pool_details()
        stock_ids = list(stock_pool_details.keys())
        results_stock = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.write("大數據批量下載運算中...")
        
        all_stock_data = yf.download(stock_ids, period="1y", group_by='ticker')
        
        for idx, ticker in enumerate(stock_ids):
            progress_bar.progress((idx + 1) / len(stock_ids))
            try:
                sub_df = all_stock_data[ticker].dropna() if isinstance(all_stock_data.columns, pd.MultiIndex) else all_stock_data.dropna()
                if sub_df.empty or len(sub_df) < 20: continue
                score, buy_p, sell_p, _, _, _, _ = calculate_signals(sub_df)
                latest_price = round(sub_df['Close'].iloc[-1], 1)
                
                results_stock.append({
                    "股票名稱": stock_pool_details[ticker]["名稱"],
                    "代號": ticker,
                    "板塊": stock_pool_details[ticker]["板塊"],
                    "現價": latest_price,
                    "信心": score,
                    "買點": buy_p,
                    "賣點": sell_p
                })
            except:
                continue
                
        status_text.write("✨ 排序完成")
        progress_bar.empty()
        
        main_df = pd.DataFrame(results_stock)
        
        if not main_df.empty:
            st.write("### 📊 1. 板塊資金強度排行")
            sector_summary = main_df.groupby("板塊")["信心"].mean().reset_index()
            sector_summary = sector_summary.sort_values(by="信心", ascending=False).reset_index(drop=True)
            sector_summary["強度(%)"] = sector_summary["信心"].round(1)
            
            st.dataframe(sector_summary[["板塊", "強度(%)"]], use_container_width=True)
            
            st.divider()
            st.write("### 🏆 2. 強勢個股 Top 20")
            res_stock_df = main_df.sort_values(by="信心", ascending=False).head(20).reset_index(drop=True)
            
            # 【完美修正】移除了原本重複輸出的全市場未篩選 DataFrame，只保留這一行精準的 Top 20 呈現
            st.dataframe(
                res_stock_df[["股票名稱", "代號", "板塊", "現價", "信心", "買點", "賣點"]], 
                use_container_width=True,
                column_config={
                    "信心": st.column_config.ProgressColumn(
                        "信心", format="%d%%", min_value=0, max_value=100
                    )
                }
            )
        else:
            st.error("未能成功生成選股矩陣。")
