# main.py
import sys
import asyncio
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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
            start_date = end_date - timedelta(days=90)
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


def add_technical_indicators(df):
    """添加技术指标"""
    # 简单移动平均线
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # 相对强弱指数 (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # 避免除零
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 布林带
    df['bollinger_mid'] = df['close'].rolling(window=20).mean()
    df['bollinger_std'] = df['close'].rolling(window=20).std()
    df['bollinger_upper'] = df['bollinger_mid'] + 2 * df['bollinger_std']
    df['bollinger_lower'] = df['bollinger_mid'] - 2 * df['bollinger_std']
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 波动率
    df['volatility_20'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
    
    # 动量指标
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
    
    # 成交量变化
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_5']
    
    # 计算真实波动幅度(ATR)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    
    return df

# -------------------------------
# 计算改进的置信度分数
# -------------------------------
def calculate_confident_score(df):
    """计算多维度的置信度分数"""
    # 1. 日内价格变化信号
    df['price_signal'] = (df['close'] - df['open']) / df['open']
    
    # 2. 相对于布林带的位置 (标准化到 -1 到 1 之间)
    df['bollinger_position'] = (2 * (df['close'] - df['bollinger_lower']) / 
                                (df['bollinger_upper'] - df['bollinger_lower'] + 1e-10) - 1)
    
    # 3. RSI 信号 (标准化到 -1 到 1)
    df['rsi_signal'] = 2 * (df['rsi'] / 100) - 1
    
    # 4. MACD信号
    max_macd = df['macd_hist'].abs().max()
    df['macd_signal_norm'] = df['macd_hist'] / (max_macd + 1e-10)
    
    # 5. 动量信号
    df['momentum_signal'] = df['momentum_5'] + df['momentum_10'] * 0.5 + df['momentum_20'] * 0.25
    
    # 6. 成交量异常信号
    df['volume_signal'] = (df['volume_ratio'] - 1) * np.sign(df['price_signal'])
    
    # 使用回归模型预测真实收益
    # 如果是实际应用，这里应该是使用真实的下一日收益作为标签
    # 在这个演示中，我们使用价格信号作为代理
    df['next_day_return'] = df['price_signal'].shift(-1)
    
    # 构建综合置信度分数 (用于模型训练)
    df['confident_score'] = (
        df['price_signal'] * 0.25 +           # 价格趋势
        df['bollinger_position'] * 0.15 +     # 布林带位置
        df['rsi_signal'] * 0.15 +             # RSI指标
        df['macd_signal_norm'] * 0.15 +       # MACD柱状图
        df['momentum_signal'] * 0.2 +         # 多时间段动量
        df['volume_signal'] * 0.1             # 成交量异常
    )
    
    # 波动率调整
    df['confidence_volatility_adjusted'] = df['confident_score'] / (df['volatility_20'] + 0.01)
    
    return df

# -------------------------------
# 数据集构造：时间序列数据集
# -------------------------------
class StockDataset(Dataset):
    def __init__(self, df, seq_length=30, predict_window=1):
        """
        df: 包含历史数据和计算出的技术指标的DataFrame
        seq_length: 序列长度
        predict_window: 预测窗口（预测未来第几天）
        """
        self.seq_length = seq_length
        self.predict_window = predict_window
        
        # 选择特征列（所有数值列，但排除时间戳和目标列）
        exclude_cols = ['timestamp', 'next_day_return', 'confident_score', 'confidence_volatility_adjusted']
        self.feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        # 确保数据足够
        if len(df) < seq_length + predict_window:
            print(f"警告: 数据行数 ({len(df)}) 小于所需的最小行数 ({seq_length + predict_window}).")
            # 创建空的特征和标签数组，但保持正确的列数
            self.features = np.zeros((0, len(self.feature_cols))).astype(np.float32)
            self.labels = np.array([]).astype(np.float32)
            # 仍然创建 scaler 以便稍后使用
            self.scaler = StandardScaler()
        else:
            # 提取特征和标签
            self.features = df[self.feature_cols].fillna(method='ffill').fillna(0).values.astype(np.float32)
            # 预测的是波动率调整后的置信度分数
            self.labels = df['confidence_volatility_adjusted'].fillna(0).values.astype(np.float32)
            
            # 标准化特征
            self.scaler = StandardScaler()
            if len(self.features) > 0:
                self.features = self.scaler.fit_transform(self.features)
        
    def __len__(self):
        return max(0, len(self.features) - self.seq_length - self.predict_window + 1)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"索引 {idx} 超出范围 (0, {len(self)-1})")
            
        x = self.features[idx:idx+self.seq_length]
        y = self.labels[idx+self.seq_length+self.predict_window-1]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# -------------------------------
# 自注意力模块
# -------------------------------
class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        return self.layer_norm(x + attn_output)

# -------------------------------
# 增强版 Transformer 回归模型
# -------------------------------
class EnhancedTransformerRegressor(nn.Module):
    def __init__(self, feature_size, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.input_linear = nn.Linear(feature_size, d_model)
        self.pos_encoder = nn.Embedding(500, d_model)  # 位置编码
        
        # 自注意力层
        self.attention_layers = nn.ModuleList([
            SelfAttention(d_model, nhead) for _ in range(num_layers)
        ])
        
        # 前馈神经网络层
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 输出层
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        src: (batch_size, seq_length, feature_size)
        """
        batch_size, seq_length = src.shape[0], src.shape[1]
        
        # 线性投影和位置编码
        x = self.input_linear(src)
        positions = torch.arange(0, seq_length, device=src.device).expand(batch_size, seq_length)
        x = x + self.pos_encoder(positions)
        x = self.dropout(x)
        
        # 自注意力层
        for attn_layer in self.attention_layers:
            x = attn_layer(x)
            x = x + self.feedforward(x)
            x = self.layer_norm(x)
        
        # 取序列的最后一个时间步
        x = x[:, -1, :]
        
        # 回归预测
        out = self.regressor(x)
        return out.squeeze(1)

# -------------------------------
# 随机森林模型（用于集成）
# -------------------------------
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# -------------------------------
# 训练函数
# -------------------------------
def train_model(model, train_loader, val_loader=None, num_epochs=50, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': []}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        val_loss = train_loss
        if val_loader:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    val_running_loss += loss.item() * X.size(0)
            val_loss = val_running_loss / len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            model.train()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
        
        scheduler.step(val_loss)
        
        # 更新进度条和状态文本
        progress = (epoch + 1) / num_epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    # 加载最佳模型（如果有验证集）
    if val_loader and best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history

# -------------------------------
# 集成预测函数
# -------------------------------
def ensemble_predict(transformer_model, rf_model, X_seq, X_latest, model_weights=[0.6, 0.4]):
    """
    组合 Transformer 和随机森林模型的预测结果
    """
    transformer_model.eval()
    with torch.no_grad():
        transformer_pred = transformer_model(X_seq).item()
    
    rf_pred = rf_model.predict([X_latest])[0]
    
    # 加权平均预测结果
    ensemble_pred = transformer_pred * model_weights[0] + rf_pred * model_weights[1]
    
    # 计算模型的置信度：基于两个模型预测的接近程度
    # 如果两个模型预测接近，置信度较高
    confidence = 1.0 - min(abs(transformer_pred - rf_pred) / (max(abs(transformer_pred), abs(rf_pred)) + 1e-10), 0.5) * 2
    
    return {
        'ensemble_prediction': ensemble_pred,
        'transformer_prediction': transformer_pred,
        'rf_prediction': rf_pred,
        'model_agreement_confidence': confidence
    }

# -------------------------------
# 结果解释函数
# -------------------------------
def interpret_confidence_score(score):
    """Interpret the meaning of the confidence score"""
    if score > 0.5:
        strength = "Strong"
        direction = "upward"
    elif score > 0.2:
        strength = "Moderate"
        direction = "upward"
    elif score > 0:
        strength = "Weak"
        direction = "upward"
    elif score > -0.2:
        strength = "Weak" 
        direction = "downward"
    elif score > -0.5:
        strength = "Moderate"
        direction = "downward"
    else:
        strength = "Strong"
        direction = "downward"

    magnitude = abs(score)
    if magnitude > 0.7:
        expected_move = "significant movement expected"
    elif magnitude > 0.4:
        expected_move = "moderate movement expected"
    else:
        expected_move = "minor movement expected"
    
    return f"{strength} {direction} signal, {expected_move}. Confidence score: {score:.4f}"


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
                seq_length = 30
                # Prepare data
                data_list = []
                for agg in aggs:
                    data_list.append({
                        "timestamp": datetime.fromtimestamp(agg.timestamp / 1000).strftime('%Y-%m-%d'),
                        "open": agg.open,
                        "high": agg.high,
                        "low": agg.low,
                        "close": agg.close,
                        "volume": agg.volume,
                        "vwap": agg.vwap if hasattr(agg, 'vwap') else None,
                        "transactions": agg.transactions if hasattr(agg, 'transactions') else None
                    })

                # Create DataFrame and sort by timestamp
                df = pd.DataFrame(data_list).sort_values(by="timestamp").reset_index(drop=True)
                print(df.head())

                # Fill missing values
                df['vwap'] = df['vwap'].fillna(df['close'])
                df['transactions'] = df['transactions'].fillna(df['volume'] / 100)

                # Compute technical indicators
                df = add_technical_indicators(df)

                # Calculate confidence scores
                df = calculate_confident_score(df)

                # Drop rows with missing values
                df = df.dropna().reset_index(drop=True)

                # Display basic stock information
                st.write(f"Data Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                # Create tabs for visualization 
                tabs = st.tabs(["Price Trend", "Technical Indicators", "Historical Confidence Score", "Raw Data"])

                # Tab 1: Stock Price Trends
                with tabs[0]:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df['timestamp'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name="Candlestick"
                    ))
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['sma_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='orange')
                    ))
                    fig.update_layout(title="Stock Price Chart", xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig, use_container_width=True)

                # Tab 2: Technical Indicators
                with tabs[1]:
                    indicator_fig = go.Figure()
                    indicator_fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['rsi'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple')
                    ))
                    indicator_fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['macd'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue')
                    ))
                    indicator_fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['macd_signal'],
                        mode='lines',
                        name='MACD Signal',
                        line=dict(color='red')
                    ))
                    indicator_fig.update_layout(title="Technical Indicators", xaxis_title="Date")
                    st.plotly_chart(indicator_fig, use_container_width=True)

                # Tab 3: Confidence Scores
                with tabs[2]:
                    confidence_fig = px.line(
                        df,
                        x="timestamp",
                        y=["confident_score", "confidence_volatility_adjusted"],
                        title="Historical Confidence Scores"
                    )
                    st.plotly_chart(confidence_fig, use_container_width=True)

                # Tab 4: Raw Data
                with tabs[3]:
                    st.dataframe(df)

                # Check if data is sufficient
                if len(df) <= seq_length:
                    st.error(f"Insufficient historical data. Need at least {seq_length + 1} days of data.")
                
                st.markdown(f"<h2>Multi-Agent Analysis</h2>", unsafe_allow_html=True)
                financial_crew = FinancialCrew(company)
                result = financial_crew.run()
                # AI Analysis
                with st.spinner("Generating AI analysis..."):
                    st.write(result.get("final_report"))
                   
                    # 时间序列交叉验证
                    tscv = TimeSeriesSplit(n_splits=5)
                    
                    # Extract feature columns
                    exclude_cols = ['timestamp', 'next_day_return', 'confident_score', 'confidence_volatility_adjusted']
                    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
                    
                    X = df[feature_cols].values
                    y = df['confidence_volatility_adjusted'].values
                    
                    # Prepare data for random forest model
                    X_train, X_test = X[:-1], X[-1:]
                    y_train = y[:-1]
                    
                    # Prepare data for Transformer model
                    dataset = StockDataset(df, seq_length=seq_length)
                    
                    # 80% training, 20% validation
                    train_size = int(0.8 * len(dataset))
                    val_size = len(dataset) - train_size
                    
                    if val_size > 0:
                        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
                        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
                    else:
                        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
                        val_loader = None
                    
                    
                    # Train Transformer model 
                    feature_size = X.shape[1]
                    transformer_model = EnhancedTransformerRegressor(feature_size)
                    transformer_model, history = train_model(
                        transformer_model, 
                        train_loader, 
                        val_loader, 
                        num_epochs=80, 
                        learning_rate=1e-3
                    )
                    
                    # Train random forest model (if ensemble is chosen)
                    rf_model = None
                    ensemble=True
                    if ensemble:
                        rf_model = train_random_forest(X_train, y_train)
                    
                    # Make predictions
                    transformer_model.eval()
                    
                    # Prepare latest sequence data for prediction - more robust method
                    try:
                        # Get last seq_length rows of original feature data
                        last_sequence = df[feature_cols].fillna(0).iloc[-seq_length:].values
                        
                        # Apply same standardization as training data
                        last_sequence = dataset.scaler.transform(last_sequence)
                        
                        # Convert to tensor and add batch dimension
                        X_latest_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)
                        
                        st.write(f"Prediction data shape: {X_latest_seq.shape}")  # Debug info
                        
                        if X_latest_seq.shape[1] == 0 or X_latest_seq.shape[2] == 0:
                            raise ValueError("Invalid prediction data dimensions")
                            
                    except Exception as e:
                        st.error(f"Error preparing prediction data: {str(e)}")
                        st.error("Will try fallback method")
                        
                        # Fallback: Create temporary prediction tensor from raw data
                        temp_features = np.zeros((1, seq_length, len(feature_cols)))
                        X_latest_seq = torch.tensor(temp_features, dtype=torch.float32)
                    
                    # Make predictions
                    prediction_results = {}
                    try:
                        with torch.no_grad():
                            transformer_pred = transformer_model(X_latest_seq).item()
                            prediction_results['transformer_prediction'] = transformer_pred
                    except Exception as e:
                        st.error(f"Model prediction error: {str(e)}")
                        # Use reasonable default if prediction fails
                        transformer_pred = 0.0
                        prediction_results['transformer_prediction'] = transformer_pred
                        st.warning("Using default prediction value of 0.0")
                        
                    if ensemble and rf_model is not None:
                        # Get latest feature vector (already standardized)
                        X_latest_features = dataset.features[-1]
                        
                        try:
                            # Get latest feature vector
                            if dataset.features.shape[0] > 0:
                                X_latest_features = dataset.features[-1]
                            else:
                                # Use zero vector if features are empty
                                X_latest_features = np.zeros(len(feature_cols))
                            
                            # Make prediction
                            prediction_results = ensemble_predict(
                                transformer_model, 
                                rf_model, 
                                X_latest_seq, 
                                X_latest_features, 
                                model_weights=[0.6, 0.4]
                            )
                        except Exception as e:
                            st.error(f"Ensemble prediction error: {str(e)}")
                            # Use default prediction results
                            prediction_results = {
                                'ensemble_prediction': transformer_pred,
                                'transformer_prediction': transformer_pred,
                                'rf_prediction': 0.0,
                                'model_agreement_confidence': 0.5
                            }
                        
                    # Display prediction results
                    st.subheader("Prediction Results")
                    
                    # Create result panels
                    result_cols = st.columns(2)
                    
                    with result_cols[0]:
                        if ensemble:
                            st.metric(
                                label="Ensemble Model Predicted Confidence Score", 
                                value=f"{prediction_results['ensemble_prediction']:.4f}"
                            )
                            
                            # Show individual model predictions
                            st.write("Individual Model Predictions:")
                            st.write(f"- Transformer: {prediction_results['transformer_prediction']:.4f}")
                            st.write(f"- Random Forest: {prediction_results['rf_prediction']:.4f}")
                            st.write(f"- Model Agreement: {prediction_results['model_agreement_confidence']:.2f}")
                            
                            # Use ensemble result as final prediction
                            final_prediction = prediction_results['ensemble_prediction']
                        else:
                            st.metric(
                                label="Transformer Predicted Confidence Score", 
                                value=f"{prediction_results['transformer_prediction']:.4f}"
                            )
                            
                            # Use Transformer result as final prediction
                            final_prediction = prediction_results['transformer_prediction']
                    
                    with result_cols[1]:
                        # Show interpretation
                        st.subheader("Prediction Interpretation")
                        st.write(interpret_confidence_score(final_prediction))
                        
                        # Calculate prediction confidence interval (simplified)
                        prediction_std = 0.2  # Fixed value, should be based on model uncertainty in practice
                        lower_bound = final_prediction - 1.96 * prediction_std
                        upper_bound = final_prediction + 1.96 * prediction_std
                        
                        st.write(f"95% Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
                        
                    # Visualize training history
                    if 'train_loss' in history and len(history['train_loss']) > 0:
                        st.subheader("Model Training History")
                        history_df = pd.DataFrame({
                            'epoch': list(range(1, len(history['train_loss']) + 1)),
                            'train_loss': history['train_loss'],
                            'val_loss': history.get('val_loss', [None] * len(history['train_loss']))
                        })
                        
                        loss_fig = px.line(
                            history_df, 
                            x='epoch', 
                            y=['train_loss', 'val_loss'],
                            title="Training Loss Curve"
                        )
                        st.plotly_chart(loss_fig, use_container_width=True)
                    
                    # Clear progress information
                    
                    st.success("Prediction completed!")

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