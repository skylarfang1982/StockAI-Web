import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 網頁基本設定
st.set_page_config(page_title="AI 量化決策與選股系統", layout="wide")
st.title("🔮 多模型 AI 綜合決策系統")

# ----------------------------------------------------
# 🗂 預設台股大型與中型成分股清單 + 類股對應表 (0050 + 0051 精選)
# ----------------------------------------------------
@st.cache_data(ttl=3600)
def get_tw_stock_pool_with_sectors():
    # 定義股票代號與其對應的產業類股，方便後續統計
    return {
        # --- 台灣 50 成分股精選 ---
        "2330.TW": "半導體", "2317.TW": "電腦週邊", "2454.TW": "半導體", "2308.TW": "電子零組件",
        "2382.TW": "電腦週邊", "2881.TW": "金融保險", "2882.TW": "金融保險", "2891.TW": "金融保險",
        "2303.TW": "半導體", "3711.TW": "半導體", "2412.TW": "通信網路", "2886.TW": "金融保險",
        "2357.TW": "電腦週邊", "3231.TW": "電腦週邊", "2324.TW": "電腦週邊", "1301.TW": "塑膠工業",
        "1303.TW": "塑膠工業", "2603.TW": "航運業", "2609.TW": "航運業", "2615.TW": "航運業",
        "2002.TW": "鋼鐵工業", "2379.TW": "半導體", "3008.TW": "光電業", "2884.TW": "金融保險",
        "2892.TW": "金融保險", "2880.TW": "金融保險", "2885.TW": "金融保險", "2887.TW": "金融保險",
        "5880.TW": "金融保險", "2890.TW": "金融保險", "2353.TW": "電腦週邊", "2371.TW": "電機機械",
        "2352.TW": "電腦週邊", "2345.TW": "通信網路", "2377.TW": "電腦週邊", "2395.TW": "電腦週邊",
        "3034.TW": "半導體", "3037.TW": "電子零組件", "3045.TW": "通信網路", "4938.TW": "電腦週邊",
        "2408.TW": "半導體", "2409.TW": "光電業", "3481.TW": "光電業", "1101.TW": "水泥工業",
        "1402.TW": "紡織纖維", "5876.TW": "金融保險", "2207.TW": "汽車工業", "9910.TW": "運動休閒",
        
        # --- 中型 100 成分股精選 (補齊中型股動能) ---
        "2327.TW": "電子零組件", "2618.TW": "航運業", "2610.TW": "航運業", "2347.TW": "電子通路",
        "2360.TW": "電子零組件", "3035.TW": "半導體", "2337.TW": "半導體", "2401.TW": "半導體",
        "2449.TW": "半導體", "2474.TW": "電腦週邊", "2354.TW": "電腦週邊", "2355.TW": "電子零組件",
        "3532.TW": "半導體", "6116.TW": "光電業", "2455.TW": "通信網路", "2458.TW": "電子通路",
        "3702.TW": "電子通路", "3005.TW": "電子通路", "2383.TW": "電子零組件", "2385.TW": "電子零組件",
        "3044.TW": "電子零組件", "6213.TW": "電子零組件", "6239.TW": "半導體", "6271.TW": "半導體",
        "5483.TW": "半導體", "3105.TW": "半導體", "6488.TW": "半導體", "1795.TW": "生技醫療",
        "4147.TW": "生技醫療", "1707.TW": "化學工業", "1722.TW": "化學工業", "2105.TW": "橡膠工業",
        "9921.TW": "運動休閒", "9914.TW": "運動休閒", "9904.TW": "製鞋業", "9945.TW": "建材營造",
        "2542.TW": "建材營造", "2548.TW": "建材營造", "2912.TW": "貿易百貨", "5904.TW": "貿易百貨",
        "2633.TW": "航運業", "2707.TW": "觀光餐旅", "8464.TW": "材料生技", "6669.TW": "電腦週邊",
        "2329.TW": "半導體", "6415.TW": "半導體", "3653.TW": "電腦週邊", "5269.TW": "半導體"
    }

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

    # 狀態計分 (最新一日數據)
    m1_buy = (df['MACD_Hist'].iloc[-1] > 0 and df['OBV'].iloc[-1] > df['OBV_MA5'].iloc[-1] and df['Close'].iloc[-1] > df['MA20'].iloc[-1])
    m1_sell = (df['MACD_Hist'].iloc[-1] < 0 or df['Close'].iloc[-1] < df['MA20'].iloc[-1])
    m1_score = 1 if m1_buy else (-1 if m1_sell else 0)

    # 短線動能優化：KD黃金交叉或RSI強勢
    m2_buy = (df['K'].iloc[-1] > df['D'].iloc[-1] and df['RSI'].iloc[-1] > 45)
    m2_sell = (df['K'].iloc[-1] < df['D'].iloc[-1] and df['RSI'].iloc[-1] > 70)
    m2_score = 1 if m2_buy else (-1 if m2_sell else 0)

    m3_buy = df['Close'].iloc[-1] >= df['BB_Upper'].iloc[-1]
    m3_sell = df['Close'].iloc[-1] <= df['BB_Lower'].iloc[-1]
    m3_score = 1 if m3_buy else (-1 if m3_sell else 0)

    # 綜合信心度計算 (只要有多個模型轉向多頭，分數就往上拉)
    positive_signals = [1 for s in [m1_score, m2_score, m3_score] if s == 1]
    confidence_score = int((len(positive_signals) / 3) * 100)
    
    # 估算買賣點
    recent_low = df['Low'].iloc[-20:].min()
    recent_high = df['High'].iloc[-20:].max()
    best_buy = round((recent_low + df['BB_Lower'].iloc[-1]) / 2, 1)
    best_sell = round((recent_high + df['BB_Upper'].iloc[-1]) / 2, 1)
    
    return confidence_score, best_buy, best_sell, m1_score, m2_score, m3_score, df

# ----------------------------------------------------
# 🌐 網頁前端分頁配置 (Tabs)
# ----------------------------------------------------
tab1, tab2 = st.tabs(["🔍 個股策略診斷", "🚀 即時潛力雷達 (0050+0051 精選)"])

# ===== Tab 1: 個股診斷功能 =====
with tab1:
    st.write("輸入特定股票代號，查看最完整的量化指標明細與動態買賣操作指引。")
    ticker_input = st.text_input("請輸入股票代號（台股如 2330.TW，美股如 NVDA）", value="2330.TW", key="single_search")
    
    if st.button("啟動個股分析"):
        with st.spinner('數據計算中...'):
            df = yf.download(ticker_input, start="2024-01-01")
            if df.empty:
                st.error("無法抓取該股票數據，請檢查代號。")
            else:
                score, buy_p, sell_p, m1, m2, m3, processed_df = calculate_signals(df)
                latest_price = round(processed_df['Close'].iloc[-1], 1)
                latest_date = processed_df.index[-1].strftime('%Y-%m-%d')
                
                st.subheader(f"📊 綜合診斷報告 (分析日期: {latest_date})")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("當前收盤價", f"{latest_price} 元")
                col2.metric("綜合看漲信心度", f"{score} %")
                
                if score >= 66:
                    col3.success("🔥 強烈建議買入 (共識度高)")
                elif score == 33:
                    col3.info("🍏 偏多續抱 / 少量試單")
                elif (m1 == -1 or m2 == -1 or m3 == -1) and score == 0:
                    col3.error("🚨 建議避險 / 減碼賣出")
                else:
                    col3.warning("💤 趨勢不明 / 觀望盤整")

                st.divider()

                # 最佳買賣價位
                st.write("### 🎯 現階段最佳動態策略價位估算")
                buy_col, sell_col, action_col = st.columns(3)
                with buy_col:
                    st.markdown("##### 🟢 最佳分批買點 (強支撐位)")
                    st.markdown(f"## **{buy_p} 元**")
                with sell_col:
                    st.markdown("##### 🔴 最佳波段賣點 (強壓力位)")
                    st.markdown(f"## **{sell_p} 元**")
                with action_col:
                    st.markdown("##### ✏️ 策略空間操作指引")
                    if latest_price <= buy_p * 1.02:
                        st.success("✨ 目前股價已極度接近「最佳買點」，適合逢低佈局！")
                    elif latest_price >= sell_p * 0.98:
                        st.error("⚠️ 目前股價已高度逼近「最佳賣點」，請勿追高，宜分批獲利入袋！")
                    else:
                        pct_to_buy = round(((latest_price - buy_p) / latest_price) * 100, 1)
                        st.info(f"當前價格處於合理震盪區間。距離下方最佳買點約有 **{pct_to_buy}%** 的修正空間。")

                st.divider()

                # 模型分項
                st.write("### 🔍 各模型獨立審查明細")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write("**1. 趨勢籌碼流 (MACD+OBV)**")
                    st.write("🟢 趨勢看多" if m1 == 1 else "🔴 趨勢看空/轉弱")
                with c2:
                    st.write("**2. 動能反轉流 (RSI+KD)**")
                    st.write("🟢 處於低檔反彈區" if m2 == 1 else ("🔴 處於高檔過熱區" if m2 == -1 else "🟡 訊號中立"))
                with c3:
                    st.write("**3. 波動突破流 (布林通道)**")
                    st.write("🟢 帶量突破上軌 (強勢股)" if m3 == 1 else ("🔴 跌破下軌 (弱勢格局)" if m3 == -1 else "🟡 通道內震盪"))

                with st.expander("📊 點擊查看歷史 K 線與波動度通道走勢圖"):
                    fig, ax = plt.subplots(figsize=(14, 5))
                    ax.plot(processed_df['Close'], label='收盤價', color='black', alpha=0.7)
                    ax.plot(processed_df['BB_Upper'], label='布林上軌', color='red', linestyle='--', alpha=0.5)
                    ax.plot(processed_df['BB_Middle'], label='布林中軌', color='orange', alpha=0.5)
                    ax.plot(processed_df['BB_Lower'], label='布林下軌', color='green', linestyle='--', alpha=0.5)
                    ax.legend(loc='upper left')
                    st.pyplot(fig)

# ===== Tab 2: 升級版選股雷達功能 (0050+0051完全體 + 類股統計) =====
with tab2:
    st.header("🎛 台股大/中型股多模型篩選器")
    st.write("系統將自動掃描包含 **台灣50 (0050)** 與 **中型100 (0051)** 在內的核心成分股池，篩選出最強的前10名個股，並統計資金流向類股。")
    
    if st.button("開始全面掃描台股市場"):
        stock_pool_dict = get_tw_stock_pool_with_sectors()
        stock_pool = list(stock_pool_dict.keys())
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.write("正在下載台股大/中型核心股票池最新數據...")
        all_data = yf.download(stock_pool, period="3mo", group_by='ticker')
        
        for idx, ticker in enumerate(stock_pool):
            status_text.write(f"系統多模型交叉運算中： {ticker} ...")
            progress_bar.progress((idx + 1) / len(stock_pool))
            
            try:
                if isinstance(all_data.columns, pd.MultiIndex):
                    sub_df = all_data[ticker].dropna()
                else:
                    sub_df = all_data.dropna()
                    
                if sub_df.empty or len(sub_df) < 20:
                    continue
                    
                score, buy_p, sell_p, _, _, _, _ = calculate_signals(sub_df)
                latest_price = round(sub_df['Close'].iloc[-1], 1)
                
                # 取得該股票對應的類股
                sector = stock_pool_dict.get(ticker, "其他")
                
                results.append({
                    "股票代號": ticker,
                    "所屬類股": sector,
                    "最新收盤價": f"{latest_price} 元",
                    "看漲信心指數": score,
                    "建議分批買點": f"{buy_p} 元",
                    "建議波段賣點": f"{sell_p} 元"
                })
            except Exception as e:
                continue
                
        status_text.write("✨ 全市場量化掃描完成！")
        progress_bar.empty()
        
        res_df = pd.DataFrame(results)
        if not res_df.empty:
            # 依據信心指數排序，取出前 10 名
            top_10_df = res_df.sort_values(by="看漲信心指數", ascending=False).head(10).reset_index(drop=True)
            
            # 📊 區塊一：顯示前10名股票明細
            st.write("### 🏆 潛力股排行榜 Top 10 (大型股 + 中型股)")
            st.dataframe(
                top_10_df, 
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
            
            st.divider()
            
            # 📊 區塊二：【新需求】前10名強勢股的類股統計
            st.write("### 🎯 熱門主流類股資金流向統計 (Top 10 類股分佈)")
            
            # 計算前 10 名中各類股出現的次數與比例
            sector_counts = top_10_df["所屬類股"].value_counts().reset_index()
            sector_counts.columns = ["所屬類股", "上榜家數"]
            sector_counts["資金聚集權重"] = (sector_counts["上榜家數"] / sector_counts["上榜家數"].sum()) * 100
            sector_counts["資金聚集權重"] = sector_counts["資金聚集權重"].astype(int)
            
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.write("下方表格顯示目前最強勢的前 10 名股票主要集中在哪些產業。上榜家數越多，代表該產業目前的「類股族群性」越整齊、漲勢越健康。")
                st.dataframe(sector_counts, use_container_width=True)
            
            with col_right:
                # 畫出漂亮的類股分佈水平條形圖
                fig, ax = plt.subplots(figsize=(6, 3))
                # 解決 matplotlib 顯示中文亂碼問題 (設定微軟正黑體或標準字體)
                plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                
                colors = plt.cm.get_cmap('Dark2')(np.linspace(0, 1, len(sector_counts)))
                ax.barh(sector_counts["所屬類股"], sector_counts["上榜家數"], color=colors, height=0.5)
                ax.set_xlabel("上榜企業家數")
                ax.set_title("前 10 名強勢股產業分佈流向")
                ax.grid(axis='x', linestyle=':', alpha=0.6)
                st.pyplot(fig)
                
            st.success("💡 資金流向判讀技巧：量化交易中「買強不買弱，選股先選族群」。如果發現『半導體』或『電子零組件』在 Top 10 中佔了 3 家以上，代表該產業正值波段主流資金浪潮，這時候進場勝率會比單打獨鬥的個股高出許多！")
        else:
            st.warning("當前市場暫未掃描到符合多指標共鳴的多頭股票。")
