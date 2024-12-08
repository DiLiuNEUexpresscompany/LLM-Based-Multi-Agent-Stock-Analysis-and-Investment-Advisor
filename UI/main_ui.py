import streamlit as st
import pandas as pd
import plotly.express as px
from polygon import RESTClient
from datetime import datetime, timedelta
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
from cachetools import cached, TTLCache
import time
import random

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Intelligent Stock Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styles
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .block-container { padding: 3rem 5rem !important; max-width: 1200px; }
    h1 { color: #1e293b; font-size: 2.25rem !important; font-weight: 700 !important; }
    .search-container, .chart-container, .analysis-container {
        background: white; padding: 1.5rem; border-radius: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .stButton > button { background-color: #1e293b !important; color: white !important; }
    .stButton > button:hover { background-color: #334155 !important; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# Initialize API clients
POLYGON_API_KEY = os.getenv("STOCK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not POLYGON_API_KEY:
    st.error("Missing Polygon API key. Please set STOCK_API_KEY in your environment variables.")
    st.stop()

if not OPENAI_API_KEY:
    st.error("Missing OpenAI API key. Please set OPENAI_API_KEY in your environment variables.")
    st.stop()

client = RESTClient(POLYGON_API_KEY)
client_OpenAI = OpenAI(api_key=OPENAI_API_KEY)

# Cache configuration
cache = TTLCache(maxsize=100, ttl=3600)  # Cache up to 100 requests with a TTL of 1 hour

def exponential_backoff(attempt):
    """Exponential backoff strategy for handling rate limit errors."""
    base_delay = 1  # Initial delay
    max_delay = 30  # Maximum delay
    delay = min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, 1)
    time.sleep(delay)

def extract_ticker_from_query(query: str) -> str:
    """从查询中提取股票代码或公司名称"""
    static_mapping = {
        "nvidia": "NVDA",
        "apple": "AAPL",
        "tesla": "TSLA",
        # 添加更多的映射
    }

    try:
        # 调用 OpenAI 模型
        response = client_OpenAI.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a stock assistant that maps company names to their stock tickers."},
                {"role": "user", "content": f"What is the stock ticker for the company mentioned in this query: '{query}'?"}
            ],
            temperature=0.3
        )
        print("OpenAI Response:", response)  # 调试信息
        
        # 从返回的文本中提取股票代码
        ticker_raw = response.choices[0].message.content.strip()
        ticker_match = re.search(r'\b[A-Z]{1,5}\b', ticker_raw)  # 提取可能的股票代码
        if ticker_match:
            return ticker_match.group(0)
        else:
            print("No valid ticker found in:", ticker_raw)
    except Exception as e:
        print(f"Error during LLM ticker extraction: {str(e)}")

    # 静态映射检查
    for company, ticker in static_mapping.items():
        if company.lower() in query.lower():
            return ticker

    # 使用正则表达式直接从用户查询中提取可能的股票代码
    ticker_match = re.search(r'\b[A-Z]{1,5}\b', query)
    if ticker_match:
        return ticker_match.group(0)

    return None
    

@cached(cache)
def get_stock_data(ticker):
    """Fetch stock data from Polygon API with retry and exponential backoff support."""
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            info = client.get_ticker_details(ticker)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            aggs = list(client.list_aggs(ticker, 1, 'day', start_date, end_date))
            return info, aggs
        except Exception as e:
            print(f"Attempt {attempt + 1} - Error fetching data for {ticker}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                if "429" in str(e):
                    st.error("API rate limit exceeded. Please try again later.")
                elif "NOT_FOUND" in str(e):
                    st.error(f"Stock symbol {ticker} not found. Please check your input.")
                else:
                    st.error(f"Error fetching stock data: {str(e)}")
            exponential_backoff(attempt)
    return None, None

def format_number(num):
    """Format large numbers, e.g., 1K, 1M, 1B."""
    if num is None:
        return "N/A"
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return f"${num:.1f}{['', 'K', 'M', 'B', 'T'][magnitude]}"

# Main application layout
st.title("Intelligent Stock Analysis")

query = st.text_input("Enter a stock-related query (e.g., 'How is Apple performing lately?')")
if st.button("Analyze"):
    with st.spinner("Processing..."):
        ticker = extract_ticker_from_query(query)
        
        if not ticker:
            st.error("Unable to extract stock ticker from the query. Please try again.")
        else:
            st.markdown(f"### {ticker}")
            info, aggs = get_stock_data(ticker)
            
            if not info or not aggs:
                st.error(f"Failed to fetch data for ticker: {ticker}. Please check the input or try again later.")
            else:
                # Display stock metrics
                cols = st.columns(4)
                metrics = [
                    ("Market Cap", format_number(info.market_cap)),
                    ("Volume", f"{aggs[-1].volume:,}"),
                    ("Closing Price", f"${aggs[-1].close:.2f}"),
                    ("Total Employees", f"{info.total_employees:,}" if info.total_employees else "N/A")
                ]
                for col, (label, value) in zip(cols, metrics):
                    with col:
                        st.metric(label, value)
                
                # Plot price chart
                df = pd.DataFrame(aggs)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                fig = px.line(df, x='timestamp', y='close', template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                # AI Analysis
                with st.spinner("Generating AI analysis..."):
                    analysis_prompt = f"""
                    Analyze the stock data for {ticker} ({info.name}):
                    - Current price: ${aggs[-1].close:.2f}
                    - 30-day price range: ${min(a.low for a in aggs):.2f} - ${max(a.high for a in aggs):.2f}
                    - Market Cap: {format_number(info.market_cap)}
                    """
                    response = client_OpenAI.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "system", "content": "You are a professional stock analyst."},
                                  {"role": "user", "content": analysis_prompt}],
                        temperature=0.7
                    )
                    st.write(response.choices[0].message.content)
                pass