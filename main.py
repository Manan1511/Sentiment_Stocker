"""
Sentiment Stocker - Stock Sentiment Analyzer Dashboard
A Streamlit application that analyzes stock sentiment based on news headlines.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from GoogleNews import GoogleNews
import random

# Initialize NLTK VADER lexicon
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Page configuration
st.set_page_config(
    page_title="Sentiment Stocker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .gradient-text {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-positive {
        color: #00ff00;
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-negative {
        color: #ff0000;
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-neutral {
        color: #808080;
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_name(ticker: str) -> str:
    """Fetch the full name of the stock."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('longName', ticker)
    except Exception:
        return ticker


@st.cache_data(ttl=3600)  # Cache for 1 hour to prevent API rate limits
def fetch_stock_data(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """
    Fetch stock price data using yfinance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')
        period: Time period for historical data (default: 1 month)
    
    Returns:
        DataFrame with stock price history
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            st.error(f"No data found for ticker: {ticker}")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_news_headlines(ticker: str, max_results: int = 10) -> list:
    """
    Fetch recent news headlines for a stock ticker.
    
    CRITICAL: News scraping often fails with 403 Forbidden errors.
    This function implements a fallback to mock headlines to prevent crashes.
    
    Args:
        ticker: Stock ticker symbol
        max_results: Maximum number of headlines to fetch
    
    Returns:
        List of headline strings
    """
    headlines = []
    
    try:
        # Remove exchange suffix for better news search (e.g., RELIANCE.NS -> RELIANCE)
        clean_ticker = ticker.split('.')[0]
        
        # Try using GoogleNews library
        googlenews = GoogleNews(lang='en', period='7d')
        googlenews.search(f'{clean_ticker} stock')
        results = googlenews.results()
        
        if results:
            headlines = [item['title'] for item in results[:max_results]]
        
        # If no results, try alternative method with requests
        if not headlines:
            url = f"https://news.google.com/rss/search?q={clean_ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                headlines = [item.title.text for item in items[:max_results]]
        
    except Exception as e:
        st.warning(f"News scraping failed: {str(e)}. Using mock data instead.")
    
    # FALLBACK MODE: Generate mock headlines if scraping failed
    if not headlines:
        mock_headlines = [
            f"{ticker} stock shows strong performance in recent trading session",
            f"Analysts upgrade {ticker} with positive outlook for next quarter",
            f"{ticker} announces new strategic initiatives to boost growth",
            f"Market volatility impacts {ticker} trading volume",
            f"{ticker} reports mixed results in latest earnings call",
            f"Investors remain cautious about {ticker} amid market uncertainty",
            f"{ticker} stock gains momentum on positive sector trends",
            f"Technical analysis suggests bullish pattern for {ticker}",
            f"{ticker} faces headwinds from regulatory concerns",
            f"Institutional investors increase holdings in {ticker}"
        ]
        headlines = random.sample(mock_headlines, min(max_results, len(mock_headlines)))
        st.info("📰 Using mock headlines for demonstration (news scraping unavailable)")
    
    return headlines


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of text using NLTK VADER.
    
    VADER returns 4 scores:
    - neg: Negative sentiment probability (0-1)
    - neu: Neutral sentiment probability (0-1)
    - pos: Positive sentiment probability (0-1)
    - compound: Normalized compound score (-1 to +1)
    
    The compound score is the most useful single metric:
    - >= 0.05: Positive sentiment
    - <= -0.05: Negative sentiment
    - Between -0.05 and 0.05: Neutral sentiment
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary with sentiment scores
    """
    scores = sia.polarity_scores(text)
    return scores


def calculate_average_sentiment(headlines: list) -> tuple:
    """
    Calculate average sentiment score across all headlines.
    
    Args:
        headlines: List of headline strings
    
    Returns:
        Tuple of (average_compound_score, sentiment_details_list)
    """
    sentiment_details = []
    total_compound = 0
    
    for headline in headlines:
        scores = analyze_sentiment(headline)
        sentiment_details.append({
            'headline': headline,
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        })
        total_compound += scores['compound']
    
    avg_sentiment = total_compound / len(headlines) if headlines else 0
    
    return avg_sentiment, sentiment_details


def create_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Create an interactive Plotly chart for stock price history.
    
    Args:
        df: DataFrame with stock price data
        ticker: Stock ticker symbol
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    fig.update_layout(
        title=f'{ticker} - Stock Price History (Last 30 Days)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_dark',
        height=500,
        showlegend=True
    )
    
    return fig


def get_sentiment_color(score: float) -> str:
    """
    Determine color based on sentiment score.
    
    Args:
        score: Compound sentiment score (-1 to +1)
    
    Returns:
        Color class name
    """
    if score > 0.05:
        return "metric-positive"
    elif score < -0.05:
        return "metric-negative"
    else:
        return "metric-neutral"


def get_sentiment_label(score: float) -> str:
    """
    Get sentiment label based on score.
    
    Args:
        score: Compound sentiment score (-1 to +1)
    
    Returns:
        Sentiment label string
    """
    if score > 0.05:
        return "🟢 Positive"
    elif score < -0.05:
        return "🔴 Negative"
    else:
        return "⚪ Neutral"


# Main Application
def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 <span class="gradient-text">Sentiment Stocker</span></h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #888;">Stock Sentiment Analysis Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Analysis Settings")
    st.sidebar.markdown("---")
    
    ticker = st.sidebar.text_input(
        "Stock Ticker Symbol",
        value="",
        help="Enter a valid stock ticker (e.g., AAPL, RELIANCE.NS, TSLA)"
    )
    
    analyze_button = st.sidebar.button("🔍 Analyze", type="primary", use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard fetches stock price data and recent news headlines, "
        "then uses NLTK's VADER sentiment analyzer to calculate sentiment scores."
    )
    
    # Main content
    if analyze_button:
        if not ticker:
            st.error("Please enter a stock ticker symbol.")
            return
        
        with st.spinner(f"Analyzing {ticker}..."):
            # Fetch stock data
            stock_data = fetch_stock_data(ticker)
            
            if stock_data is None or stock_data.empty:
                st.error("Unable to fetch stock data. Please check the ticker symbol.")
                return
            
            # Fetch stock name
            stock_name = get_stock_name(ticker)
            
            # Fetch news headlines
            headlines = fetch_news_headlines(ticker)
            
            if not headlines:
                st.error("Unable to fetch news headlines.")
                return
            
            # Calculate sentiment
            avg_sentiment, sentiment_details = calculate_average_sentiment(headlines)
            
            # Get current price
            current_price = stock_data['Close'].iloc[-1]
            
            # Display Stock Name
            st.markdown(f"<h2 style='text-align: center; margin-bottom: 20px;'>{stock_name} ({ticker})</h2>", unsafe_allow_html=True)
            
            # TOP SECTION: Display metrics
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="📈 Current Stock Price",
                    value=f"${current_price:.2f}",
                    delta=f"{((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-2] - 1) * 100):.2f}%"
                )
            
            with col2:
                sentiment_color = get_sentiment_color(avg_sentiment)
                sentiment_label = get_sentiment_label(avg_sentiment)
                st.markdown(f"**Average Sentiment Score**")
                st.markdown(f'<p class="{sentiment_color}">{avg_sentiment:.3f}</p>', unsafe_allow_html=True)
                st.markdown(f"**{sentiment_label}**")
            
            with col3:
                st.metric(
                    label="📰 Headlines Analyzed",
                    value=len(headlines)
                )
            
            # MIDDLE SECTION: Stock price chart
            st.markdown("---")
            st.subheader("📊 Stock Price History")
            fig = create_price_chart(stock_data, ticker)
            st.plotly_chart(fig, use_container_width=True)
            
            # BOTTOM SECTION: Headlines and sentiment scores
            st.markdown("---")
            st.subheader("📰 News Headlines & Sentiment Analysis")
            
            # Display headlines as individual cards to avoid pyarrow dependency
            for i, detail in enumerate(sentiment_details, 1):
                score = detail['compound']
                label = '🟢 Positive' if score > 0.05 else ('🔴 Negative' if score < -0.05 else '⚪ Neutral')
                
                # Create a colored box for each headline
                col_a, col_b, col_c = st.columns([6, 1, 1])
                with col_a:
                    st.markdown(f"**{i}.** {detail['headline']}")
                with col_b:
                    st.markdown(f"**{score:.3f}**")
                with col_c:
                    st.markdown(f"{label}")
                
                if i < len(sentiment_details):
                    st.markdown("---")

            
            # Additional insights
            st.markdown("---")
            st.subheader("💡 Sentiment Insights")
            
            positive_count = len([s for s in sentiment_details if s['compound'] > 0.05])
            negative_count = len([s for s in sentiment_details if s['compound'] < -0.05])
            neutral_count = len(sentiment_details) - positive_count - negative_count
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🟢 Positive Headlines", positive_count)
            with col2:
                st.metric("🔴 Negative Headlines", negative_count)
            with col3:
                st.metric("⚪ Neutral Headlines", neutral_count)
    
    else:
        # Welcome message
        st.info("👈 Enter a stock ticker in the sidebar and click 'Analyze' to get started!")
        
        st.markdown("### How it works:")
        st.markdown("""
        1. **Enter a stock ticker** (e.g., AAPL for Apple, RELIANCE.NS for Reliance Industries)
        2. **Click Analyze** to fetch stock data and news headlines
        3. **View sentiment analysis** powered by NLTK's VADER algorithm
        4. **Explore interactive charts** showing price history and sentiment breakdown
        """)
        
        st.markdown("### Example Tickers:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- **AAPL** - Apple Inc.")
            st.markdown("- **TSLA** - Tesla Inc.")
            st.markdown("- **GOOGL** - Alphabet Inc.")
        with col2:
            st.markdown("- **RELIANCE.NS** - Reliance Industries (India)")
            st.markdown("- **TCS.NS** - Tata Consultancy Services (India)")
            st.markdown("- **INFY.NS** - Infosys (India)")


if __name__ == "__main__":
    main()