# main.py
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
from crews.crew import FinancialCrew


from agents.tool_registry import ToolRegistry
from agents.visualization_agent import VisualizationAgent

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
    /* Define custom variables */
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Playfair+Display:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
    :root {
        --primary-100:#3F51B5;
        --primary-200:#757de8;
        --primary-300:#dedeff;
        --accent-100:#2196F3;
        --accent-200:#003f8f;
        --text-100:#333333;
        --text-200:#5c5c5c;
        --bg-100:#FFFFFF;
        --bg-200:#f5f5f5;
        --bg-300:#cccccc;
    }
        

    /* Main background and text */
    body {
        background-color: var(--bg-100) !important;
        color: var(--text-100) !important;
        font-family: 'Cormorant Garamond', serif;
    }

    /* Main container for Streamlit */
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-100) !important;
        color: var(--text-100) !important;
    }

    /* Sidebar container (if applicable) */
    [data-testid="stSidebar"] {
        background-color: var(--bg-200) !important;
        color: var(--text-100) !important;
    }

    /* Content container */
    .block-container {
        padding: 3.5rem 6rem !important;
        max-width: 1300px;
    }

    /* Headings */
    h1 {
        font-family: 'Playfair Display', serif !important;
        color: var(--primary-100) !important;
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        border-bottom: 2px solid var(--primary-100);
        padding-bottom: 0.5rem;
    }
    h2 {
        font-family: 'Playfair Display', serif !important;
        color: var(--primary-100) !important; 
        font-size: 2.0rem !important;
        font-weight: 700 !important; 
        letter-spacing: 0.5px !important;
        border-bottom: 2px solid var(--primary-100); 
        padding-bottom: 0.5rem;
        margin-bottom: 1rem; 
    }
    h3 {
        font-family: 'Playfair Display', serif !important;
        color: var(--primary-100) !important; 
        font-size: 1.5rem !important;
        font-weight: 700 !important; 
        letter-spacing: 0.5px !important;
        border-bottom: 2px solid var(--primary-100); 
        padding-bottom: 0.5rem;
        margin-bottom: 1rem; 
    }
    h4 {
        font-family: 'Playfair Display', serif !important;
        color: var(--primary-100) !important; 
        font-size: 1.0rem !important;
        font-weight: 700 !important; 
        letter-spacing: 0.5px !important;
        border-bottom: 2px solid var(--primary-100); 
        padding-bottom: 0.5rem;
        margin-bottom: 1rem; 
    }        

    /* Stock header styles */
    .stock-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.5rem 0;
    }

    .stock-code {
        font-family: 'DM Mono', monospace;
        font-size: 1.5rem;
        font-weight: 500;
        color: var(--primary-200);
    }

    .stock-icon {
        color: var(--accent-100);
        font-size: 1.2rem;
    }

    /* Metric card styles */
    .metric-container {
        background-color: var(--primary-200);
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 5px;
        color: var(--text-100);
    }

    .metric-label {
        font-family: 'Playfair Display', serif !important;
        font-size: 18px;
        color: var(--bg-100);
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: var(--primary-300);
    }

    /* Button styles */
    .stButton > button {
        background-color: var(--primary-100) !important;
        color: var(--bg-200) !important;
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 1.1rem !important;
        letter-spacing: 1px;
        padding: 0.5rem 2rem !important;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: var(--accent-200) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Chart container styles */
    .chart-container {
        background: var(--bg-300);
        padding: 2rem;
        border-radius: 0.5rem;
        border: 1px solid var(--bg-200);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 2rem 0;
    }

    /* Input and text areas */
    textarea, input {
        background-color: var(--bg-100) !important;
        color: var(--text-100) !important;
        border: 2px solid var(--primary-200) !important;
    }
    

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

static_mapping = {
    "Nvidia": "NVDA",
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Meta": "META"
    # 添加更多的映射
}

def extract_ticker_and_company_from_query(query: str) -> tuple:
    """
    从查询中提取公司名称和股票代码。
    优先查找映射，未找到则调用模型，并更新映射。
    """
    global static_mapping  # 使用全局映射

    # 检查映射中是否存在
    for company, ticker in static_mapping.items():
        if company.lower() in query.lower():
            return company, ticker

    try:
        # 调用 OpenAI 模型获取公司名称
        company_response = client_OpenAI.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a stock assistant that extracts company names from a query."},
                {"role": "user", "content": f"What is the company name mentioned in this query: '{query}.'? Just give me it's name."}
            ],
            temperature=0.3
        )
        print("OpenAI Company Response:", company_response)  # 调试信息

        company_name = company_response.choices[0].message.content.strip()
        company_name = re.sub(r"[^A-Za-z\s]", "", company_name)  # 移除非字母字符
        
        # 验证是否是有效公司名称
        if not company_name or len(company_name.split()) > 3 or company_name.lower() in ["the", "a", "an"]:
            company_name = None

        # 调用 OpenAI 模型获取股票代码
        ticker_response = client_OpenAI.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a stock assistant that extracts stock tickers from a query."},
                {"role": "user", "content": f"What is the stock ticker for the company mentioned in this query: '{query}.'? Just give me it's name."}
            ],
            temperature=0.3
        )
        print("OpenAI Ticker Response:", ticker_response)  # 调试信息

        ticker_raw = ticker_response.choices[0].message.content.strip()
        ticker_match = re.search(r'\b[A-Z]{1,5}\b', ticker_raw)  # 提取可能的股票代码
        ticker = ticker_match.group(0) if ticker_match else None

        # 如果公司名称和股票代码都有效，更新映射并返回结果
        if company_name and ticker:
            static_mapping[company_name.lower()] = ticker  # 更新映射
            print(f"Mapping updated: {company_name} -> {ticker}")
            return company_name, ticker

    except Exception as e:
        print(f"Error during LLM extraction: {str(e)}")

    # 如果静态映射和模型都未能提供结果，返回默认值
    return None, None

    

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

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Sidebar setup
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-content {
        background-color: var(--bg-200);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .history-title {
        color: var(--primary-100);
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-100);
    }
    .history-item {
        background-color: var(--bg-300);
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .query-text {
        color: var(--text-100);
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .ticker-text {
        color: var(--primary-200);
        font-family: 'DM Mono', monospace;
        font-size: 1rem;
        font-weight: 500;
    }
    .timestamp-text {
        color: var(--text-200);
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="history-title">Conversation History</p>', unsafe_allow_html=True)
    
    if len(st.session_state.conversation_history) == 0:
        st.markdown('<div class="sidebar-content">No history yet</div>', unsafe_allow_html=True)
    else:
        for item in reversed(st.session_state.conversation_history):
            st.markdown(f"""
            <div class="history-item">
                <div class="query-text">{item['query']}</div>
                <div class="ticker-text">{item['ticker']}</div>
                <div class="timestamp-text">{item['timestamp']}</div>
            </div>
            """, unsafe_allow_html=True)

# Main content
st.title("Intelligent Stock Analysis")

query = st.text_input("Enter a stock-related query (e.g., 'How is Apple performing lately?')")
if st.button("Analyze"):
    with st.spinner("Processing..."):
        company, ticker = extract_ticker_and_company_from_query(query)
        
        if not ticker:
            st.error("Unable to extract stock ticker from the query. Please try again.")
        else:
            # Add to conversation history
            from datetime import datetime
            st.session_state.conversation_history.append({
                'query': query,
                'ticker': ticker,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.markdown(f"<h3>{ticker}</h3>", unsafe_allow_html=True)
            info, aggs = get_stock_data(ticker)
            
            if not info or not aggs:
                st.error(f"Failed to fetch data for ticker: {ticker}. Please check the input or try again later.")
            else:
                # Rest of your existing code remains the same
                cols = st.columns(4)
                metrics = [
                    ("Market Cap", format_number(info.market_cap)),
                    ("Volume", f"{aggs[-1].volume:,}"),
                    ("Closing Price", f"${aggs[-1].close:.2f}"),
                    ("Total Employees", f"{info.total_employees:,}" if info.total_employees else "N/A")
                ]
                for col, (label, value) in zip(cols, metrics):
                    with col:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)
                                
                # Plot price chart
                df = pd.DataFrame(aggs)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                fig = px.line(df, x='timestamp', y='close', 
                            template="plotly_dark",  # Using dark template as base
                            line_shape='spline')

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Close Price",
                    font=dict(
                        family="Arial, sans-serif",
                        size=16,
                        color="#333333"
                    ),
                    xaxis=dict(
                        title=dict(
                            font=dict(
                                size=18,
                            )
                        ),
                        tickfont=dict(
                            size=14,
                        ),
                    ),
                    yaxis=dict(
                        title=dict(
                            font=dict(
                                size=18,
                            )
                        ),
                        tickfont=dict(
                            size=14,
                        ),
                    )
                )

                # Update line color and properties
                fig.update_traces(
                    line_color='#3F51B5',     # --primary-100
                    line_width=2,
                    hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
                )

                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"<h2>Multi-Agent Analysis</h2>", unsafe_allow_html=True)
                financial_crew = FinancialCrew(company)
                result = financial_crew.run()
                # AI Analysis
                with st.spinner("Generating AI analysis..."):
                    st.write(result.get("final_report"))
                    registry = ToolRegistry()
                    visualization_agent = VisualizationAgent(registry)
                    visualizations = visualization_agent.run(result)

                    # 在 streamlit 中显示图表
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.plotly_chart(visualizations["confidence"], use_container_width=True)
                    with col2:
                        st.plotly_chart(visualizations["growth"], use_container_width=True)
                    with col3:
                        st.plotly_chart(visualizations["radar"], use_container_width=True)


                with st.expander("Click to view all retrieved files"):
                    recent_news = result.get("news_data")
                    price_trend = "\n" + result.get("price_data")
                    report_analysis = "Here are the Report Analysis: \n" + result.get("report_data")
                    st.markdown(
                        f"""
                        <div style="color: black;">
                            <h2 style="color: black;">Recent News: </h2>
                            <p>{recent_news}</p>
                            <h2 style="color: black;">Price Trend: </h2>
                            <p>{price_trend}</p>
                            <h2 style="color: black;">Report Analysis: </h2>
                            <p>{report_analysis}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )