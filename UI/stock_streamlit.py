import streamlit as st
import pandas as pd
import plotly.express as px
from polygon import RESTClient
from datetime import datetime, timedelta

# Initialize session state variables
if 'stock_types' not in st.session_state:
    st.session_state.stock_types = ""
if 'exchanges' not in st.session_state:
    st.session_state.exchanges = ""

# Streamlit app details
st.set_page_config(page_title="Financial Analysis", layout="wide")
with st.sidebar:
    st.title("Financial Analysis")
    ticker = st.text_input("Stock ticker (e.g. AAPL)", "AAPL")
    polygon_api_key = st.text_input("Polygon API key", type="password")
    button = st.button("Submit")

# Authenticate with the Polygon API
client = RESTClient(polygon_api_key)

# Get stock ticker type description for a given code
@st.cache_data
def get_stock_type(code):
    for stock_type in st.session_state.stock_types:
        if stock_type.code == code:
            return stock_type.description
    return None

# Get stock exchange name for a given code
@st.cache_data
def get_exchange_name(code):
    for exchange in st.session_state.exchanges:
        if exchange.mic == code:
            return exchange.name
    return None

# Format market cap and financial info into readable values
@st.cache_data
def format_value(value):
    suffixes = ["", "K", "M", "B", "T"]
    suffix_index = 0
    while value >= 1000 and suffix_index < len(suffixes) - 1:
        value /= 1000
        suffix_index += 1
    return f"${value:.1f}{suffixes[suffix_index]}"

# If Submit button is clicked
if button:
    if not polygon_api_key.strip():
        st.error("Please provide a valid API key.")
    elif not ticker.strip():
        st.error("Please provide a valid stock ticker.")
    else:
        try:
            with st.spinner('Please wait...'):
                # Get supported stock ticker types
                if not st.session_state.stock_types:
                    st.session_state.stock_types = client.get_ticker_types(asset_class='stocks')

                # Get supported stock exchanges
                if not st.session_state.exchanges:
                    st.session_state.exchanges = client.get_exchanges(asset_class='stocks')

                # Retrieve stock ticker details
                info = client.get_ticker_details(ticker)
                st.subheader(f"{ticker} - {info.name}")

                # Plot historical price chart for the last 30 days
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=30)

                history = client.list_aggs(ticker, 1, 'day', start_date, end_date, limit=50)
                chart_data = pd.DataFrame(history) 
                chart_data['timestamp'] = pd.to_datetime(chart_data['timestamp'], unit='ms') 
                chart_data['date'] = chart_data['timestamp'].dt.strftime('%Y-%m-%d')
                
                price_chart = px.line(chart_data, x='date', y='close', width=1000, height=400, line_shape='spline')
                price_chart.update_layout(
                    xaxis_title="Date",         
                    yaxis_title="Price"
                )

                st.plotly_chart(price_chart)

                col1, col2, col3 = st.columns(3)

                # Display stock information as a dataframe
                stock_info = [
                    ("Stock Info", "Value"),
                    ("Type", get_stock_type(info.type)),
                    ("Primary Exchange", get_exchange_name(info.primary_exchange)),
                    ("Listing Date", info.list_date),
                    ("Market Cap", format_value(info.market_cap)),
                    ("Employees", "{:,.0f}".format(info.total_employees)),
                    ("Website", info.homepage_url.replace("https://", ""))
                ]
                
                df = pd.DataFrame(stock_info[1:], columns=stock_info[0])
                col1.dataframe(df, width=400, hide_index=True)
                
                # Display price information as a dataframe
                agg = client.get_previous_close_agg(ticker)

                price_info = [
                    ("Price Info", "Value"),
                    ("Prev Day Close", f"${agg[0].close:.2f}"),
                    ("Prev Day Open", f"${agg[0].open:.2f}"),
                    ("Prev Day High", f"${agg[0].high:.2f}"),
                    ("Prev Day Low", f"${agg[0].low:.2f}"),
                    ("Volume", "{:,.0f}".format(agg[0].volume)),
                    ("VW Avg Price", f"${agg[0].vwap:.2f}")
                ]
                
                df = pd.DataFrame(price_info[1:], columns=price_info[0])
                col2.dataframe(df, width=400, hide_index=True)
                
                # Display historical financial information as a dataframe
                fin = client.vx.list_stock_financials(ticker, sort='filing_date', order='desc', limit=2)
                
                for item in fin:
                    break
                
                fin_metrics = [
                    ("Financial Metrics", "Value"),
                    ("Fiscal Period", item.fiscal_period + " " + item.fiscal_year),
                    ("Total Assets", format_value(item.financials.balance_sheet['assets'].value)),
                    ("Total Liabilities", format_value(item.financials.balance_sheet['liabilities'].value)),
                    ("Revenues", format_value(item.financials.income_statement.revenues.value)),
                    ("Net Cash Flow", format_value(item.financials.cash_flow_statement.net_cash_flow.value)),
                    ("Basic EPS", f"${item.financials.income_statement.basic_earnings_per_share.value}")
                ]
                
                df = pd.DataFrame(fin_metrics[1:], columns=fin_metrics[0])
                col3.dataframe(df, width=400, hide_index=True)

        except Exception as e:
            if "too many 429 error responses" in str(e):
                st.error("Max retries exceeded! Please upgrade your Polygon API plan or wait for a while...")
            else:
                st.exception(f"An error occurred: {e}")