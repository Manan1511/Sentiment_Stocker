"""
Sentiment Stocker v7.8 - Refined Layout
Features: 
- Clean Sidebar (Controls Only)
- Fundamentals moved to Main View (PEG Ratio Removed)
- Topic Modeling (The "Why")
- Supply Chain Ripple Effects
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
import random
from collections import Counter
from io import BytesIO
import numpy as np
import re

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Sentiment Stocker | Pro Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

# --- 2. DATA CONSTANTS ---

COMPETITORS = {
    'AAPL': ['MSFT', 'GOOGL', 'NVDA'],
    'MSFT': ['AAPL', 'GOOGL', 'AMZN'],
    'GOOGL': ['MSFT', 'META', 'AMZN'],
    'NVDA': ['AMD', 'INTC', 'TSM'],
    'TSLA': ['F', 'GM', 'TM'],
    'TCS.NS': ['INFY.NS', 'WIPRO.NS', 'HCLTECH.NS'],
    'INFY.NS': ['TCS.NS', 'WIPRO.NS', 'TECHM.NS'],
    'RELIANCE.NS': ['ADANIENT.NS', 'ONGC.NS', 'TATASTEEL.NS'],
    'HDFCBANK.NS': ['SBIN.NS', 'ICICIBANK.NS', 'AXISBANK.NS'],
    'TATAMOTORS.NS': ['MARUTI.NS', 'M&M.NS', 'ASHOKLEY.NS']
}

# Feature 2: Map Sectors to Supply Chain Dependencies
SECTOR_MAP = {
    'Technology': ['Semiconductors', 'Cloud Computing', 'AI Chips'],
    'Consumer Electronics': ['Semiconductors', 'Lithium'],
    'Auto': ['Steel', 'Semiconductors', 'Crude Oil'],
    'Energy': ['Crude Oil', 'Natural Gas', 'OPEC'],
    'Financial': ['Interest Rates', 'Housing Market'],
    'Healthcare': ['Biotech', 'Insurance'],
    'Utilities': ['Natural Gas', 'Coal']
}

# Feature 1: Keywords for Topic Modeling
TOPIC_KEYWORDS = {
    "⚠️ Regulatory": ['sec', 'lawsuit', 'fine', 'ban', 'court', 'investigation', 'compliance', 'judge', 'antitrust'],
    "💰 Earnings": ['eps', 'revenue', 'profit', 'earnings', 'quarterly', 'miss', 'beat', 'guidance', 'dividend'],
    "📦 Product": ['launch', 'reveal', 'defect', 'recall', 'iphone', 'model', 'upgrade', 'delay', 'chip'],
    "🌍 Macro": ['inflation', 'fed', 'rate', 'tax', 'jobs', 'recession', 'policy', 'economy', 'interest'],
    "🤝 M&A": ['acquire', 'merger', 'buyout', 'deal', 'takeover', 'stake']
}

# --- 3. CUSTOM STYLING (UI FIXES) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }
    
    /* Modern Card Container */
    .metric-container {
        background-color: #1A1C24;
        border: 1px solid #2E303E;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        color: white;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        border-color: #667eea;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #A0A0A0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 5px;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #A0A0A0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #667eea;
        border-bottom: 2px solid #667eea;
    }

    /* News Feed Styling */
    .news-card {
        background-color: #1A1C24;
        border: 1px solid #2E303E;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        transition: background-color 0.3s;
    }
    .news-card:hover {
        background-color: #252836;
    }
    .news-link {
        text-decoration: none;
        color: #E0E0E0;
        font-weight: 600;
        font-size: 1.05rem;
    }
    .news-meta {
        font-size: 0.8rem;
        color: #888;
        margin-top: 5px;
    }
    
    /* Main Title */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. CORE FUNCTIONS ---

def clean_google_link(url):
    if not url: return "#"
    if "news.google.com" in url or "google.com/url" in url: return url 
    if "&ved=" in url: return url.split("&ved=")[0]
    return url

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="1mo"):
    try:
        stock = yf.Ticker(ticker)
        # Fetch slightly more data for SMA calculations
        hist_period = '1y' if period in ['1mo', '3mo'] else period
        df = stock.history(period=hist_period)
        return df, stock.info
    except: return None, None

@st.cache_data(ttl=1800)
def fetch_news_data(query, limit=15):
    """Modified to accept general query strings for Topic/Sector analysis"""
    news_data = []
    try:
        googlenews = GoogleNews(lang='en', period='7d')
        googlenews.search(query)
        results = googlenews.results()
        if results:
            for item in results[:limit]:
                news_data.append({
                    'title': item['title'],
                    'link': clean_google_link(item.get('link', '#')),
                    'source': item.get('media', 'Google News'),
                    'date': item.get('date', 'Recent')
                })
    except: pass
    
    return news_data

def get_sentiment_score(headlines):
    total = 0
    for h in headlines:
        total += sia.polarity_scores(h['title'])['compound']
    return total / len(headlines) if headlines else 0

def categorize_topics(news_items):
    """Feature 1: Tags news with specific topics"""
    tags = []
    for n in news_items:
        title = n['title'].lower()
        found_tag = False
        for category, keywords in TOPIC_KEYWORDS.items():
            if any(k in title for k in keywords):
                tags.append(category)
                found_tag = True
                break
        if not found_tag:
            tags.append("📰 General News")
    return Counter(tags)

# --- 5. UI COMPONENTS ---

def render_kpi(label, value, delta=None):
    delta_html = f'<div class="metric-delta" style="color:{"#00E676" if "+" in delta else "#FF1744"};">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def render_news_card(title, link, source, score):
    color = "#00E676" if score > 0.05 else ("#FF1744" if score < -0.05 else "#B0BEC5")
    st.markdown(f"""
    <div class="news-card" style="border-left: 4px solid {color};">
        <a href="{link}" target="_blank" rel="noreferrer noopener" class="news-link">
            {title}
        </a>
        <div class="news-meta">
            <span>{source}</span> • <span style="color:{color}; font-weight:bold;">Score: {score:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_ripple_effects(sector):
    """Feature 2: Supply Chain Ripple Effects Component"""
    st.markdown(f"#### ⛓️ Supply Chain: {sector} Dependencies")
    st.caption("Monitoring upstream sectors for early warning signals")
    
    dependencies = SECTOR_MAP.get(sector, ['Global Markets', 'Oil'])
    cols = st.columns(len(dependencies))
    
    for i, dep in enumerate(dependencies):
        with cols[i]:
            dep_news = fetch_news_data(f"{dep} market news", limit=5)
            score = get_sentiment_score(dep_news)
            
            color = "#00E676" if score > 0.05 else "#FF1744" if score < -0.05 else "#A0A0A0"
            arrow = "⬆" if score > 0.05 else "⬇" if score < -0.05 else "➡"
            
            st.markdown(f"""
            <div style="background:#1A1C24; padding:15px; border-radius:10px; border:1px solid #2E303E; text-align:center;">
                <div style="color:#888; font-size:0.8rem; text-transform:uppercase;">{dep}</div>
                <div style="color:{color}; font-size:1.5rem; font-weight:bold;">{score:.2f} {arrow}</div>
            </div>
            """, unsafe_allow_html=True)

def plot_advanced_timeline(df, current_sentiment, period):
    # Slice data for view
    if period == '1mo': view_df = df.tail(30)
    elif period == '6mo': view_df = df.tail(180)
    else: view_df = df

    # SMAs
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    plot_df = df.loc[view_df.index]

    fig = go.Figure()
    
    # Candlestick or Line? Let's go Line for cleanliness with SMAs
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name='Price', line=dict(color='#667eea', width=2)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_50'], name='SMA 50', line=dict(color='#FFA500', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_200'], name='SMA 200', line=dict(color='#FF4500', width=1, dash='dot')))
    
    # Sentiment Overlay
    days = len(plot_df)
    noise = np.random.normal(0, 0.1, days)
    trend = np.linspace(0, current_sentiment, days) + noise
    
    fig.add_trace(go.Bar(
        x=plot_df.index, y=trend, name='Sentiment Trend',
        yaxis='y2', marker_color=np.where(trend>0, '#00E676', '#FF1744'),
        opacity=0.15
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title="Price (₹)", showgrid=True, gridcolor='#2E303E'),
        yaxis2=dict(title="Sentiment", overlaying='y', side='right', range=[-1, 1], showgrid=False),
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.02, x=0)
    )
    return fig

# --- 6. MARKET MAP ---
def render_heatmap():
    st.markdown("### 🌍 Global Market Sentiment")
    # Demo Data for Visual
    data = [
        {'Ticker': 'AAPL', 'Sector': 'Tech', 'Vol': 100, 'Score': 0.6},
        {'Ticker': 'MSFT', 'Sector': 'Tech', 'Vol': 90, 'Score': 0.4},
        {'Ticker': 'NVDA', 'Sector': 'Tech', 'Vol': 120, 'Score': 0.8},
        {'Ticker': 'GOOGL', 'Sector': 'Tech', 'Vol': 80, 'Score': -0.2},
        {'Ticker': 'TSLA', 'Sector': 'Auto', 'Vol': 110, 'Score': -0.4},
        {'Ticker': 'AMZN', 'Sector': 'Tech', 'Vol': 95, 'Score': 0.2},
        {'Ticker': 'XOM', 'Sector': 'Energy', 'Vol': 85, 'Score': 0.7},
    ]
    df = pd.DataFrame(data)
    
    fig = px.treemap(df, path=[px.Constant("Global Market"), 'Sector', 'Ticker'], values='Vol', color='Score',
                     color_continuous_scale='RdYlGn', range_color=[-1, 1])
    
    fig.update_layout(height=600, margin=dict(t=0, l=0, r=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    return

# --- 7. MAIN APP ---

def main():
    # Sidebar (Purely Control Panel)
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        mode = st.radio("View Mode", ["🔍 Analysis", "🗺️ Market Heatmap"], index=0)
        st.markdown("---")
        
        if mode == "🔍 Analysis":
            ticker = st.text_input("Ticker Symbol", "AAPL").upper()
            period = st.selectbox("Timeframe", ["1mo", "6mo", "1y", "5y"])
            st.caption("Press Enter to update")
        else:
            ticker = None
    
    # Header
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<div class="main-header">Sentiment Stocker ⚡</div>', unsafe_allow_html=True)
        st.caption("Advanced Financial Intelligence Terminal")
    
    # MODE: MARKET HEATMAP
    if mode == "🗺️ Market Heatmap":
        render_heatmap()
        return

    # MODE: ANALYSIS
    if ticker:
        # Load Data
        df, info = fetch_stock_data(ticker, period)
        news = fetch_news_data(f"{ticker} stock") # Updated to search generic string
        
        if df is None:
            st.error("Stock not found. Check ticker symbol.")
            return

        # KPIs
        curr_price = df['Close'].iloc[-1]
        try:
            prev = df['Close'].iloc[-2]
            delta = f"{((curr_price-prev)/prev)*100:+.2f}%"
        except: delta = "0%"
        
        sent_score = get_sentiment_score(news)
        
        # --- TOP KPI ROW ---
        k1, k2, k3, k4 = st.columns(4)
        with k1: render_kpi("Current Price", f"₹{curr_price:.2f}", delta)
        with k2: render_kpi("Sentiment", f"{sent_score:.2f}", "(-1 to +1)")
        with k3: render_kpi("News Volume", str(len(news)), "Headlines")
        with k4: 
            mcap = info.get('marketCap', 0) if info else 0
            render_kpi("Market Cap", f"₹{mcap/1e9:.1f}B", None)

        st.write("") # Spacer

        # --- TABS LAYOUT ---
        tab1, tab2, tab3 = st.tabs(["📈 Chart & Technicals", "🧠 Sentiment Lab", "📰 News & Peers"])

        # TAB 1: CHART
        with tab1:
            st.plotly_chart(plot_advanced_timeline(df, sent_score, period), use_container_width=True)
            
            # Fundamentals Row (Moved here from Sidebar)
            st.markdown("### 🏢 Key Fundamentals")
            if info:
                f1, f2, f3 = st.columns(3)
                f1.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
                f2.metric("52W High", f"₹{info.get('fiftyTwoWeekHigh', 0):.2f}")
                f3.metric("52W Low", f"₹{info.get('fiftyTwoWeekLow', 0):.2f}")
            else:
                st.warning("No fundamental data.")

        # TAB 2: SENTIMENT DEEP DIVE (UPDATED)
        with tab2:
            # Topic Modeling Feature
            topic_counts = categorize_topics(news)
            
            c_narrative, c_ripple = st.columns([1, 1])
            
            with c_narrative:
                st.subheader("🗣️ Narrative Analysis (The 'Why')")
                if topic_counts:
                    df_topics = pd.DataFrame.from_dict(topic_counts, orient='index', columns=['Count']).reset_index()
                    fig_topic = px.pie(df_topics, values='Count', names='index', hole=0.5, 
                                       color_discrete_sequence=px.colors.sequential.RdBu)
                    fig_topic.update_layout(height=300, margin=dict(t=0,b=0))
                    st.plotly_chart(fig_topic, use_container_width=True)
                else:
                    st.info("Not enough data to determine topics.")

            with c_ripple:
                # Supply Chain Feature
                sector = info.get('sector', 'Technology') if info else 'Technology'
                render_ripple_effects(sector)

        # TAB 3: NEWS & PEERS
        with tab3:
            c_peers, c_news = st.columns([1, 2])
            
            with c_peers:
                st.subheader("⚔️ Competitors")
                peers = COMPETITORS.get(ticker, ['AAPL', 'MSFT', 'GOOGL'])[:3]
                for p in peers:
                    p_news = fetch_news_data(f"{p} stock", limit=5)
                    p_score = get_sentiment_score(p_news)
                    p_color = "#00E676" if p_score > 0.05 else "#FF1744"
                    
                    st.markdown(f"""
                    <div style="background:#262730; padding:15px; border-radius:8px; margin-bottom:10px; border-left:4px solid {p_color};">
                        <div style="font-weight:bold; font-size:1.1rem;">{p}</div>
                        <div style="color:#A0A0A0; font-size:0.9rem;">Sentiment: <span style="color:{p_color}">{p_score:.2f}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

            with c_news:
                st.subheader("Recent Headlines")
                for n in news:
                    score = sia.polarity_scores(n['title'])['compound']
                    render_news_card(n['title'], n['link'], n['source'], score)

        # Excel Export (Bottom)
        st.markdown("---")
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            pd.DataFrame(news).to_excel(writer, sheet_name='News', index=False)
            df_export = df.copy()
            if hasattr(df_export.index, 'tz'): df_export.index = df_export.index.tz_localize(None)
            for col in df_export.columns:
                if pd.api.types.is_datetime64_any_dtype(df_export[col]):
                    try: df_export[col] = df_export[col].dt.tz_localize(None)
                    except: pass
            df_export.to_excel(writer, sheet_name='Price_Data')
            
        st.download_button("📥 Download Report", buffer.getvalue(), f"{ticker}_Report.xlsx")

if __name__ == "__main__":
    main()