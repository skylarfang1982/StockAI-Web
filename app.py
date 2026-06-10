import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 網頁基本設定
st.set_page_config(page_title="AI 量化決策與選股系統", layout="wide")
st.title("🔮 多模型 AI 綜合決策系統")

# ----------------------------------------------------
# 🗂 150檔核心股池 與 其所屬產業板塊 完美映射字典
# ----------------------------------------------------
def get_stock_sector_mapping():
    mapping = {
        # --- 半導體與 IC 設計 ---
        "2330.TW": "半導體科技", "2454.TW": "半導體科技", "2303.TW": "半導體科技", "3711.TW": "半導體科技",
        "2379.TW": "半導體科技", "3034.TW": "半導體科技", "2408.TW": "半導體科技", "2337.TW": "半導體科技",
        "2449.TW": "半導體科技", "3532.TW": "半導體科技", "3035.TW": "半導體科技", "5269.TW": "半導體科技",
        "6415.TW": "半導體科技", "5347.TW": "半導體科技", "6147.TW": "半導體科技", "8299.TW": "半導體科技",
        # --- AI伺服器、電腦與電子代工 ---
        "2317.TW": "AI電腦與代工", "2382.TW": "AI電腦與代工", "2357.TW": "AI電腦與代工", "3231.TW": "AI電腦與代工",
        "2324.TW": "AI電腦與代工", "2353.TW": "AI電腦與代工", "2371.TW": "AI電腦與代工", "2352.TW": "AI電腦與代工",
        "2345.TW": "AI電腦與代工", "2377.TW": "AI電腦與代工", "2395.TW": "AI電腦與代工", "4938.TW": "AI電腦與代工",
        "6669.TW": "AI電腦與代工", "2354.TW": "AI電腦與代工", "2355.TW": "AI電腦與代工", "2376.TW": "AI電腦與代工",
        "3017.TW": "AI電腦與代工", "3264.TW": "AI電腦與代工", "3596.TW": "AI電腦與代工",
        # --- 建材營造板塊 ---
        "2542.TW": "建材營造傳統", "2548.TW": "建材營造傳統", "9945.TW": "建材營造傳統", "2501.TW": "建材營造傳統",
        "2534.TW": "建材營造傳統",
        # --- 電機機械與重電綠能 ---
        "1504.TW": "電機重電綠能", "1513.TW": "電機重電綠能", "1519.TW": "電機重電綠能", "1605.TW": "電機重電綠能",
        "9958.TW": "電機重電綠能",
        # --- 航運、航空與物流 ---
        "2603.TW": "航運航空物流", "2609.TW": "航運航空物流", "2615.TW": "航運航空物流", "2618.TW": "航運航空物流",
        "2610.TW": "航運航空物流", "2633.TW": "航運航空物流", "2605.TW": "航運航空物流", "2606.TW": "航運航空物流",
        # --- 金控與銀行 ---
        "2881.TW": "金融保險金控", "2882.TW": "金融保險金控", "2891.TW": "金融保險金控", "2886.TW": "金融保險金控",
        "2884.TW": "金融保險金控", "2892.TW": "金融保險金控", "2880.TW": "金融保險金控", "2885.TW": "金融保險金控",
        "2887.TW": "金融保險金控", "5880.TW": "金融保險金控", "2890.TW": "金融保險金控", "5876.TW": "金融保險金控",
        "2801.TW": "金融保險金控", "2809.TW": "金融保險金控", "2812.TW": "金融保險金控", "2834.TW": "金融保險金控",
        "2838.TW": "金融保險金控", "2845.TW": "金融保險金控", "2855.TW": "金融保險金控", "2883.TW": "金融保險金控",
        "2888.TW": "金融保險金控", "2889.TW": "金融保險金控",
        # --- 電子零組件、ABF、PCB與通路 ---
        "2308.TW": "電子零組件通路", "3037.TW": "電子零組件通路", "2327.TW": "電子零組件通路", "2360.TW": "電子零組件通路",
        "2474.TW": "電子零組件通路", "3702.TW": "電子零組件通路", "2383.TW": "電子零組件通路", "2385.TW": "電子零組件通路",
        "3044.TW": "電子零組件通路", "6213.TW": "電子零組件通路", "6269.TW": "電子零組件通路", "3036.TW": "電子零組件通路",
        "4958.TW": "電子零組件通路", "8046.TW": "電子零組件通路",
        # --- 生技醫療與化學 ---
        "1795.TW": "生技醫療化學", "4147.TW": "生技醫療化學", "1707.TW": "生技醫療化學", "1722.TW": "生技醫療化學",
        "1717.TW": "生技醫療化學",
        # --- 光電與面板 ---
        "2409.TW": "光電面板面板", "3481.TW": "光電面板面板", "3008.TW": "光電面板面板", "6116.TW": "光電面板面板",
        "3406.TW": "光電面板面板",
        # --- 傳統工業（水泥、鋼鐵、塑化、基礎建設） ---
        "1101.TW": "傳統基礎工業", "1402.TW": "傳統基礎工業", "2002.TW": "傳統基礎工業", "1301.TW": "傳統基礎工業",
        "1303.TW": "傳統基礎工業", "2105.TW": "傳統基礎工業", "2106.TW": "傳統基礎工業", "9933.TW": "傳統基礎工業",
        # --- 觀光、百貨、運動與內需消費 ---
        "2207.TW": "內需消費生活", "9910.TW": "內需消費生活", "2912.TW": "內需消費生活", "5904.TW": "內需消費生活",
        "2707.TW": "內需消費生活", "9921.TW": "內需消費生活", "9914.TW": "內需消費生活", "9904.TW": "內需消費生活",
        "2201.TW": "內需消費生活", "2204.TW": "內需消費生活"
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
    
    recent_low = df['Low'].iloc[-20:].min()
    recent_high = df['High'].iloc[-20:].max()
    best_buy = round((recent_low + df['BB_Lower'].iloc[-1]) / 2, 1)
    best_sell = round((recent_high + df['BB_Upper'].iloc[-1]) / 2, 1)
    
    return confidence_score, best_buy, best_sell, m1_score, m2_score, m3_score, df

# ----------------------------------------------------
# 🌐 網頁前端分頁配置
# ----------------------------------------------------
tab1, tab2 = st.tabs(["🔍 個股策略診斷", "🚀 全台股整體產業金流與強勢股雷達"])

# ===== Tab 1: 個股診斷功能 (🎯 雙階段按鈕觸發圖表) =====
with tab1:
    st.write("輸入特定股票代號，查看最完整的量化指標明細與動態買賣操作指引。")
    
    ticker_input = st.text_input("請輸入股票代號（台股如：2330.TW、2603.TW）", value="2330.TW")
    
    # 建立兩個按鈕並排的空間
    btn_col1, btn_col2 = st.columns([1, 4])
    
    with btn_col1:
        run_analysis = st.button("啟動個股分析")
    
    # 🎯 初始化用於控制「是否顯示圖表」的狀態開關
    if "show_chart" not in st.session_state:
        st.session_state.show_chart = False
    if "current_ticker" not in st.session_state:
        st.session_state.current_ticker = ""

    # 如果使用者更換了股票代號，自動把圖表收起來，直到再次點擊
    if ticker_input != st.session_state.current_ticker:
        st.session_state.show_chart = False
        st.session_state.current_ticker = ticker_input

    # 當按下主分析鈕，執行量化運算並把數據存入暫存，同時隱藏圖表
    if run_analysis:
        if not ticker_input:
            st.warning("請先輸入股票代號。")
        else:
            with st.spinner('數據計算中...'):
                df = yf.download(ticker_input, start="2024-01-01")
                if df.empty:
                    st.error("無法抓取該股票數據。")
                else:
                    # 將計算結果暫存在 session_state 中，避免網頁刷新時資料不見
                    score, buy_p, sell_p, m1, m2, m3, processed_df = calculate_signals(df)
                    st.session_state.analysis_data = {
                        "score": score, "buy_p": buy_p, "sell_p": sell_p,
                        "latest_price": round(processed_df['Close'].iloc[-1], 1),
                        "latest_date": processed_df.index[-1].strftime('%Y-%m-%d'),
                        "chart_df": processed_df.tail(65)
                    }
                    # 預設此時圖表仍不顯示
                    st.session_state.show_chart = False

    # 顯示暫存的分析數據
    if "analysis_data" in st.session_state:
        data = st.session_state.analysis_data
        
        st.subheader(f"📊 綜合診斷報告 (分析日期: {data['latest_date']})")
        col1, col2, col3 = st.columns(3)
        col1.metric("當前收盤價", f"{data['latest_price']} 元")
        col2.metric("綜合看漲信心度", f"{data['score']} %")
        if data['score'] >= 66:
            col3.success("🔥 強烈建議買入")
        elif data['score'] == 33:
            col3.info("🍏 偏多續抱 / 少量試單")
        else:
            col3.warning("💤 趨勢不明 / 觀望盤整")

        st.divider()
        st.write("### 🎯 現階段最佳動態策略價位估算")
        buy_col, sell_col, action_col = st.columns(3)
        buy_col.metric("🟢 最佳分批買點 (強支撐)", f"{data['buy_p']} 元")
        sell_col.metric("🔴 最佳波段賣點 (強壓力)", f"{data['sell_p']} 元")
        
        with action_col:
            if data['latest_price'] <= data['buy_p'] * 1.02:
                st.success("✨ 股價接近最佳買點，適合逢低佈局！")
            elif data['latest_price'] >= data['sell_p'] * 0.98:
                st.error("⚠️ 股價逼近最佳賣點，請勿追高！")
            else:
                st.info("當前價格處於合理震盪區間。")

        st.divider()
        
        # 🎯 核心設計：放上第二個專門用來控制圖表開關的按鈕
        if st.session_state.show_chart:
            if st.button("👁️ 隱藏技術對照圖表"):
                st.session_state.show_chart = False
                st.rerun()
        else:
            if st.button("📊 顯示量化買賣對照圖表"):
                st.session_state.show_chart = True
                st.rerun()

        # 🎯 如果狀態為真，才動態渲染 Matplotlib 圖表
        if st.session_state.show_chart:
            chart_df = data["chart_df"]
            buy_p = data["buy_p"]
            sell_p = data["sell_p"]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 畫出收盤價與MA20
            ax.plot(chart_df.index, chart_df['Close'], label='當日收盤價', color='#1f77b4', linewidth=2)
            ax.plot(chart_df.index, chart_df['MA20'], label='20日生命線 (MA20)', color='#ff7f0e', linestyle='--')
            
            # 動態打上買點、賣點的水平虛線
            ax.axhline(y=buy_p, color='green', linestyle=':', linewidth=1.5, label=f'最佳分批買點 ({buy_p}元)')
            ax.axhline(y=sell_p, color='red', linestyle=':', linewidth=1.5, label=f'最佳波段賣點 ({sell_p}元)')
            
            ax.set_title(f"{ticker_input} 技術趨勢與量化買賣點對照", fontsize=11, fontweight='bold')
            ax.set_xlabel("日期")
            ax.set_ylabel("價格 (元)")
            ax.legend(loc='upper left')
            ax.grid(True, linestyle=':', alpha=0.6)
            
            st.pyplot(fig)

# ===== Tab 2: 選股雷達功能 =====
with tab2:
    st.header("🎛 150大中型核心股池群組化 - 產業資金流向選股器")
    st.write("本頁面採用**混合大數據邏輯**：下載 150 檔成分股後，自動按行業分類計算並顯示數據報表。")
    
    if st.button("開始全面掃描 150 檔核心市場"):
        stock_mapping = get_stock_sector_mapping()
        stock_pool = list(stock_mapping.keys())
        results_stock = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.write("正在批量下載並運算 150 檔大中型核心股量化指標...")
        
        all_stock_data = yf.download(stock_pool, period="3mo", group_by='ticker')
        
        for idx, ticker in enumerate(stock_pool):
            progress_bar.progress((idx + 1) / len(stock_pool))
            try:
                sub_df = all_stock_data[ticker].dropna() if isinstance(all_stock_data.columns, pd.MultiIndex) else all_stock_data.dropna()
                if sub_df.empty or len(sub_df) < 20: continue
                score, buy_p, sell_p, _, _, _, _ = calculate_signals(sub_df)
                latest_price = round(sub_df['Close'].iloc[-1], 1)
                
                results_stock.append({
                    "股票代號": ticker,
                    "所屬板塊": stock_mapping[ticker],
                    "最新收盤價": f"{latest_price} 元",
                    "看漲信心指數": score,
                    "建議分批買點": f"{buy_p} 元",
                    "建議波段賣點": f"{sell_p} 元"
                })
            except:
                continue
                
        status_text.write("✨ 數據抓取完畢，正在進行辨識與群組排序...")
        progress_bar.empty()
        
        main_df = pd.DataFrame(results_stock)
        
        if not main_df.empty:
            st.write("### 📊 1. 150檔核心股池混合計算：整體產業板塊資金強度")
            sector_summary = main_df.groupby("所屬板塊")["看漲信心指數"].mean().reset_index()
            sector_summary = sector_summary.sort_values(by="看漲信心指數", ascending=False).reset_index(drop=True)
            sector_summary["板塊資金引燃強度 (%)"] = sector_summary["看漲信心指數"].round(1)
            
            st.dataframe(sector_summary[["所屬板塊", "板塊資金引燃強度 (%)"]], use_container_width=True)
            
            st.divider()
            st.write("### 🏆 2. 當前最具動能之強勢個股 Top 10 (由 150 檔核心大池篩選)")
            res_stock_df = main_df.sort_values(by="看漲信心指數", ascending=False).head(10).reset_index(drop=True)
            
            st.dataframe(
                res_stock_df[["股票代號", "所屬板塊", "最新收盤價", "看漲信心指數", "建議分批買點", "建議波段賣點"]], 
                use_container_width=True,
                column_config={
                    "看漲信心指數": st.column_config.ProgressColumn(
                        "看漲信心指數", help="多指標共鳴度", format="%d%%", min_value=0, max_value=100
                    )
                }
            )
        else:
            st.error("未能成功生成選股矩陣。")
