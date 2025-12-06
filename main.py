
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
import finnhub
import random
from collections import Counter
from io import BytesIO
import numpy as np
import re
import time
import requests
from bs4 import BeautifulSoup

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Sentiment Stocker | Pro Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DISCLAIMER MODAL (MUST COME BEFORE ANY OTHER UI) ---
if 'disclaimer_accepted' not in st.session_state:
    st.session_state.disclaimer_accepted = False

if not st.session_state.disclaimer_accepted:
    st.markdown("""
    <style>
    .disclaimer-box {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border: 2px solid #ff4b4b;
        border-radius: 15px;
        padding: 40px;
        margin: 50px auto;
        max-width: 800px;
        box-shadow: 0 10px 40px rgba(255, 75, 75, 0.3);
    }
    .disclaimer-title {
        color: #ff4b4b;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .disclaimer-subtitle {
        color: #ffaa00;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 10px;
    }
    .disclaimer-text {
        color: #e0e0e0;
        font-size: 1rem;
        line-height: 1.8;
        margin-bottom: 15px;
    }
    .warning-icon {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disclaimer-box">
        <div class="warning-icon">⚠️</div>
        <div class="disclaimer-title">SEBI Mandatory Guidelines</div>
        
        Educational Purpose Only
          
            This platform does not provide any tips, recommendations, or financial advice.
            All updates, posts, and discussions are purely and solely for educational and learning purposes
            Consult a Professional Financial Advisor before making any investment decisions.
            No Liability for Losses
            Neither the Platform Admins nor the Users are responsible for any financial losses arising from decisions made based on platform content.
            Admins will not be responsible for any financial losses incurred from transactions or dealings with any member providing buy/sell calls or claiming higher returns.
            By clicking "I Agree", you acknowledge that you have read, understood, and accept all terms and risks mentioned above.
    """, unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("I Agree and Accept the Risks", type="primary", use_container_width=True):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    
    st.stop()  # Prevent rest of app from loading

# --- REST OF YOUR CODE CONTINUES BELOW ---

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

# Initialize Finnhub Client
# Get your free API key from: https://finnhub.io/register
FINNHUB_API_KEY = "d4ps1npr01qjpnb18tdgd4ps1npr01qjpnb18te0"
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

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
    'TATAMOTORS.NS': ['MARUTI.NS', 'M&M.NS', 'ASHOKLEY.NS'],
    'TATASTEEL.NS': ['JSWSTEEL.NS', 'HINDALCO.NS', 'SAIL.NS']
}

# Feature 2: Map Sectors to Supply Chain Dependencies
SECTOR_MAP = {
    'Technology': ['Semiconductors', 'Cloud Computing', 'AI Chips'],
    'Consumer Electronics': ['Semiconductors', 'Lithium'],
    'Auto': ['Steel', 'Semiconductors', 'Crude Oil'],
    'Energy': ['Crude Oil', 'Natural Gas', 'OPEC'],
    'Financial': ['Interest Rates', 'Housing Market'],
    'Healthcare': ['Biotech', 'Insurance'],
    'Utilities': ['Natural Gas', 'Coal'],
    'Basic Materials': ['Iron Ore', 'Coal', 'Shipping']
}

DEPENDENCY_TICKERS = {
    'Semiconductors': 'SMH', 'Cloud Computing': 'SKYY', 'AI Chips': 'SOXX',
    'Lithium': 'LIT', 'Steel': 'SLX', 'Crude Oil': 'USO',
    'Natural Gas': 'UNG', 'OPEC': 'XLE', 'Interest Rates': '^TNX',
    'Housing Market': 'XHB', 'Biotech': 'IBB', 'Insurance': 'KIE',
    'Coal': 'ARCH', 'Global Markets': '^GSPC', 'Oil': 'USO',
    'Iron Ore': 'RIO', 'Shipping': 'BOAT' # BOAT is generic, maybe BDRY? Let's use BDRY or just RIO/VALE for Iron Ore
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
        height: 140px; /* Fixed height for uniformity */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        border-color: #667eea;
    }
    /* News Feed Styling */
    .news-card {
        background-color: #1A1C24;
        border: 1px solid #2E303E;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        transition: background-color 0.3s;
        min-height: 100px; /* Minimum height for uniformity */
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
        
        # USE yf.download() as it is more robust than stock.history() currently
        df = yf.download(ticker, period=hist_period, progress=False)
        
        # Flatten multi-index columns if present (common in new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty: return None, None
        return df, stock.info
    except Exception as e:
        print(f"Error fetching stock data: {e}") 
        return None, None

@st.cache_data(ttl=300)
def fetch_news_data_finnhub(ticker, limit=15):
    """Fetch financial news using Finnhub API"""
    news_data = []
    try:
        # Get date range (last 7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Format dates for Finnhub (YYYY-MM-DD)
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # Fetch company news
        news = finnhub_client.company_news(ticker, _from=from_date, to=to_date)
        
        for item in news[:limit]:
            news_data.append({
                'title': item['headline'],
                'link': item['url'],
                'source': item['source'],
                'date': datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d %H:%M'),
                'summary': item.get('summary', ''),
                'sentiment': item.get('sentiment', 0)
            })
        
        time.sleep(0.1)  # Small delay to stay under rate limit
        
    except Exception as e:
        pass  # Fail silently and return empty list
    
    return news_data

@st.cache_data(ttl=1800)
def fetch_news_data(query, limit=15):
    """Modified to accept general query strings for Topic/Sector analysis"""
    news_data = []
    # Strategy 1: YFinance (Best for "Credible News" if query is a ticker)
    # Check if query looks like a ticker (e.g., AAPL, RELIANCE.NS, ^GSPC, BRK-B)
    if not " " in query and len(query) < 15:
        try:
            t = yf.Ticker(query)
            yf_news = t.news
            if yf_news:
                for n in yf_news:
                    # Handle nested structure (yfinance update)
                    title = n.get('title')
                    link = n.get('link')
                    publisher = n.get('publisher')
                    
                    if not title and 'content' in n:
                        c = n['content']
                        title = c.get('title')
                        # Try to find link in content
                        if not link:
                            link = c.get('canonicalUrl', {}).get('url') if isinstance(c.get('canonicalUrl'), dict) else c.get('clickThroughUrl', {}).get('url')
                        # Try to find publisher in content
                        if not publisher:
                            publisher = c.get('provider', {}).get('displayName') if isinstance(c.get('provider'), dict) else "Yahoo Finance"

                    if not title: continue # Skip if no title
                    
                    ts = n.get('providerPublishTime', 0)
                    date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else 'Recent'
                    news_data.append({
                        'title': title,
                        'link': link or f"https://finance.yahoo.com/news/{n.get('id', '')}.html",
                        'source': publisher or 'Yahoo Finance',
                        'date': date_str
                    })
                return news_data[:limit]
        except: pass

    # Strategy 2: Google News (Fallback & Complex Queries like "MSFT reddit")
    # UPDATED: If query contains "reddit", try direct Reddit scraping text search fallback
    if "reddit" in query.lower():
        try:
             # Extract ticker from query (e.g., "AAPL reddit" -> "AAPL")
            search_term = query.replace("reddit", "").strip()
            # Simple reddit search scrape (old reddit is easier to parse)
            url = f"https://old.reddit.com/search?q={search_term}&sort=new"
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                # Find search results
                results = soup.find_all('div', class_='search-result-link')
                for res in results[:limit]:
                    title_tag = res.find('a', class_='search-title')
                    if title_tag:
                        news_data.append({
                            'title': title_tag.text,
                            'link': title_tag['href'] if title_tag['href'].startswith('http') else f"https://reddit.com{title_tag['href']}",
                            'source': 'Reddit',
                            'date': 'Recent'
                        })
                if news_data: return news_data
        except Exception as e:
            print(f"Reddit scrape error: {e}")

    try:
        time.sleep(2)  # Rate limiting
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
    except Exception as e:
        print(f"Google News error: {e}")
    
    return news_data

def get_sentiment_score(news_items):
    """Calculate sentiment using Finnhub sentiment or VADER fallback"""
    if not news_items:
        return 0
    
    total = 0
    for item in news_items:
        # Try using Finnhub sentiment first
        if 'sentiment' in item and item['sentiment'] != 0:
            total += item['sentiment']
        else:
            # Fallback to VADER on title + summary
            text = f"{item.get('title', '')} {item.get('summary', '')}"
            if text.strip():
                total += sia.polarity_scores(text)['compound']
    
    return total / len(news_items) if news_items else 0

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

def render_news_card(title, link, source, score, date):
    color = "#00E676" if score > 0.05 else ("#FF1744" if score < -0.05 else "#B0BEC5")
    st.markdown(f"""
    <div class="news-card" style="border-left: 4px solid {color};">
        <a href="{link}" target="_blank" rel="noreferrer noopener" class="news-link">
            {title}
        </a>
        <div class="news-meta">
            <span>{source} • {date}</span> • <span style="color:{color}; font-weight:bold;">Score: {score:.2f}</span>
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
            # Use specific ticker if available, otherwise fallback to text search (which might fail)
            query_term = DEPENDENCY_TICKERS.get(dep, dep)
            dep_news = fetch_news_data(query_term, limit=5)
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
            
        st.markdown("---")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
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
        
        # Try Finnhub first, fallback to multi-source strategy
        news = fetch_news_data_finnhub(ticker, limit=15)
        if not news:
            news = fetch_news_data(f"{ticker}")  # Fallback to yfinance/google/reddit
        
        if df is None:
            st.error("Stock data not found. Check ticker symbol or try a different timeframe.")
            return

        # KPIs
        curr_price = df['Close'].iloc[-1]
        try:
            prev = df['Close'].iloc[-2]
            delta = f"{((curr_price-prev)/prev)*100:+.2f}%"
        except: delta = "0%"
        
        sent_score = get_sentiment_score(news)
        
        # Ticker Name Display
        long_name = info.get('longName', ticker) if info else ticker
        st.markdown(f"<h3 style='text-align: center; color: #4F8BF9; margin-bottom: 20px;'>{long_name}</h3>", unsafe_allow_html=True)

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
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart & Technicals", "🧠 Sentiment Lab", "📰 News & Peers", "🔗 URL Scanner"])

        # TAB 1: CHART
        with tab1:
            st.plotly_chart(plot_advanced_timeline(df, sent_score, period), use_container_width=True)
            
            # Fundamentals Row (Moved here from Sidebar)
            st.markdown("### 🏢 Key Fundamentals")
            if info:
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
                f2.metric("52W High", f"₹{info.get('fiftyTwoWeekHigh', 0):.2f}")
                f3.metric("52W Low", f"₹{info.get('fiftyTwoWeekLow', 0):.2f}")
                f4.metric("P/B Ratio", f"{info.get('priceToBook', 0):.2f}")
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
                    # Try Finnhub first for competitors too
                    p_news = fetch_news_data_finnhub(p, limit=5)
                    if not p_news:
                        p_news = fetch_news_data(f"{p}", limit=5)
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
                
                nt1, nt2 = st.tabs(["📰 Credible News", "💬 Social Buzz"])
                
                with nt1:
                    for n in news:
                        # Calculate individual sentiment properly
                        if 'sentiment' in n and n['sentiment'] != 0:
                            # Use Finnhub's sentiment score
                            score = n['sentiment']
                        else:
                            # Fallback to VADER analysis on title + summary
                            text = f"{n.get('title', '')} {n.get('summary', '')}"
                            score = sia.polarity_scores(text)['compound'] if text.strip() else 0
                        render_news_card(n['title'], n['link'], n['source'], score, n.get('date', 'Recent'))
                        
                with nt2:
                    st.caption(f"Showing discussions for {ticker} from Reddit, Forums, etc.")
                    # Fetch "Social" news
                    social_news = fetch_news_data(f"{ticker} reddit", limit=10)
                    if social_news:
                        for n in social_news:
                            text = f"{n.get('title', '')} {n.get('summary', '')}"
                            score = sia.polarity_scores(text)['compound'] if text.strip() else 0
                            render_news_card(n['title'], n['link'], n['source'], score, n.get('date', 'Recent'))
                    else:
                        st.info(f"No substantial social buzz found for {ticker} (Reddit scraping may be blocked).")

        # TAB 4: URL SENTIMENT SCANNER
        with tab4:
            st.subheader("🔗 URL Sentiment Scanner")
            st.caption("Paste any article link to analyze its specific sentiment.")
            
            url_col1, url_col2 = st.columns([3, 1])
            with url_col1:
                url_input = st.text_input("Article URL", placeholder="https://example.com/article...", key="url_scanner_in")
            with url_col2:
                st.write("") # Alignment spacer
                analyze_btn = st.button("Analyze URL", key="url_scanner_btn")
            
            if analyze_btn and url_input:
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                    response = requests.get(url_input, headers=headers, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        # Extract paragraphs; crude but effective for general articles
                        paragraphs = soup.find_all('p')
                        text_content = " ".join([p.get_text() for p in paragraphs])
                        
                        if len(text_content) > 100:
                            url_score = sia.polarity_scores(text_content)['compound']
                            u_color = "#00E676" if url_score > 0.05 else "#FF1744" if url_score < -0.05 else "#A0A0A0"
                            u_label = "POSITIVE" if url_score > 0.05 else "NEGATIVE" if url_score < -0.05 else "NEUTRAL"
                            
                            st.markdown(f'''
                            <div style="background:#262730; padding:20px; border-radius:10px; border-left:5px solid {u_color};">
                                <h3 style="margin:0; color:white;">Sentiment: <span style="color:{u_color}">{u_label}</span> ({url_score:.2f})</h3>
                                <p style="color:#A0A0A0; margin-top:5px;">Analyzed {len(text_content)} characters from the article.</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.error("Could not extract enough text from this URL. It might be behind a paywall or Javascript-rendered.")
                    else:
                        st.error(f"Failed to fetch URL (Status Code: {response.status_code})")
                except Exception as e:
                    st.error(f"Error processing URL: {str(e)}")



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