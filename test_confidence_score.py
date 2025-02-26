import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from polygon import RESTClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from cachetools import cached, TTLCache
from main import exponential_backoff  # 使用已有的指数退避函数

# 加载环境变量，设置缓存和 Polygon API
load_dotenv()
cache = TTLCache(maxsize=100, ttl=3600)
POLYGON_API_KEY = os.getenv("STOCK_API_KEY")
client = RESTClient(POLYGON_API_KEY)

@cached(cache)
def get_stock_data(ticker):
    """使用 Polygon API 获取股票详情及历史日线数据（获取过去 180 天数据用于更好的模型训练）"""
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            info = client.get_ticker_details(ticker)
            end_date = datetime.now().date()
            # 获取更长的历史数据以便更好地计算技术指标
            start_date = end_date - timedelta(days=180)
            aggs = list(client.list_aggs(ticker, 1, 'day', start_date, end_date))
            return info, aggs
        except Exception as e:
            st.error(f"Attempt {attempt + 1} - Error fetching data for {ticker}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                if "429" in str(e):
                    st.error("API rate limit exceeded. Please try again later.")
                elif "NOT_FOUND" in str(e):
                    st.error(f"Stock symbol {ticker} not found. Please check your input.")
                else:
                    st.error(f"Error fetching stock data: {str(e)}")
            exponential_backoff(attempt)
    return None, None

# -------------------------------
# 技术指标计算函数
# -------------------------------
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
    """解释置信度分数的含义"""
    if score > 0.5:
        strength = "强烈"
        direction = "上涨"
    elif score > 0.2:
        strength = "中等"
        direction = "上涨"
    elif score > 0:
        strength = "弱"
        direction = "上涨"
    elif score > -0.2:
        strength = "弱"
        direction = "下跌"
    elif score > -0.5:
        strength = "中等"
        direction = "下跌"
    else:
        strength = "强烈"
        direction = "下跌"
    
    magnitude = abs(score)
    if magnitude > 0.7:
        expected_move = "可能出现显著波动"
    elif magnitude > 0.4:
        expected_move = "可能出现中等波动"
    else:
        expected_move = "可能波动较小"
    
    return f"{strength}{direction}信号，{expected_move}。置信度分数：{score:.4f}"

# -------------------------------
# Streamlit 主程序
# -------------------------------
def main():
    st.title("增强版股票 Confident Score 预测系统")
    
    with st.sidebar:
        st.subheader("模型参数设置")
        ticker = st.text_input("请输入股票代码（例如 AAPL）", value="AAPL")
        seq_length = st.slider("序列长度（天数）", min_value=10, max_value=60, value=30)
        epochs = st.slider("训练轮次", min_value=20, max_value=100, value=50)
        ensemble = st.checkbox("使用模型集成", value=True)
    
    if st.button("获取数据并预测"):
        # 显示进度信息
        progress_container = st.empty()
        progress_container.info("正在获取股票数据...")
        
        info, aggs = get_stock_data(ticker)
        if info is None or not aggs:
            st.error("未能获取数据，请检查股票代码或稍后重试。")
            return
        
        # 将 aggs 数据转换为 DataFrame
        data_list = []
        for agg in aggs:
            data_list.append({
                "timestamp": datetime.fromtimestamp(agg.timestamp/1000).strftime('%Y-%m-%d'),
                "open": agg.open,
                "high": agg.high,
                "low": agg.low,
                "close": agg.close,
                "volume": agg.volume,
                "vwap": agg.vwap if hasattr(agg, 'vwap') else None,
                "transactions": agg.transactions if hasattr(agg, 'transactions') else None
            })
        
        df = pd.DataFrame(data_list).sort_values(by="timestamp").reset_index(drop=True)
        
        # 填充可能的缺失值
        df['vwap'] = df['vwap'].fillna(df['close'])
        df['transactions'] = df['transactions'].fillna(df['volume'] / 100)
        
        progress_container.info("计算技术指标...")
        
        # 添加技术指标
        df = add_technical_indicators(df)
        
        # 计算增强版置信度分数
        df = calculate_confident_score(df)
        
        # 数据预处理
        df = df.dropna().reset_index(drop=True)
        
        # 显示基本数据统计
        st.subheader("股票基本信息")
        st.write(f"股票名称: {info.name}")
        st.write(f"市场: {info.market}")
        st.write(f"数据时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
        
        # 可视化股价和技术指标
        tabs = st.tabs(["股价走势", "技术指标", "历史置信度分数", "原始数据"])
        
        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="K线"
            ))
            fig.add_trace(go.Scatter(
                x=df['timestamp'], 
                y=df['sma_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange')
            ))
            fig.update_layout(title="股价K线图", xaxis_title="日期", yaxis_title="价格")
            st.plotly_chart(fig, use_container_width=True)
            
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
            indicator_fig.update_layout(title="技术指标", xaxis_title="日期")
            st.plotly_chart(indicator_fig, use_container_width=True)
            
        with tabs[2]:
            confidence_fig = px.line(
                df, 
                x="timestamp", 
                y=["confident_score", "confidence_volatility_adjusted"],
                title="历史置信度分数"
            )
            st.plotly_chart(confidence_fig, use_container_width=True)
            
        with tabs[3]:
            st.dataframe(df)
            
        # 检查数据是否足够
        if len(df) <= seq_length:
            st.error(f"历史数据不足，需要至少 {seq_length+1} 天的数据。")
            return
        
        # 数据集划分
        progress_container.info("准备训练数据...")
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 提取特征列
        exclude_cols = ['timestamp', 'next_day_return', 'confident_score', 'confidence_volatility_adjusted']
        feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        X = df[feature_cols].values
        y = df['confidence_volatility_adjusted'].values
        
        # 准备随机森林模型的数据
        X_train, X_test = X[:-1], X[-1:]
        y_train = y[:-1]
        
        # 准备 Transformer 模型的数据
        dataset = StockDataset(df, seq_length=seq_length)
        
        # 80% 训练, 20% 验证
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        if val_size > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        else:
            train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
            val_loader = None
        
        progress_container.info("开始训练模型...")
        
        # 训练 Transformer 模型
        feature_size = X.shape[1]
        transformer_model = EnhancedTransformerRegressor(feature_size)
        transformer_model, history = train_model(
            transformer_model, 
            train_loader, 
            val_loader, 
            num_epochs=epochs, 
            learning_rate=1e-3
        )
        
        # 训练随机森林模型 (如果选择集成)
        rf_model = None
        if ensemble:
            progress_container.info("训练随机森林模型...")
            rf_model = train_random_forest(X_train, y_train)
        
        # 预测
        progress_container.info("生成预测结果...")
        transformer_model.eval()
        
        # 准备最新的序列数据用于预测 - 更健壮的方法
        try:
            # 获取原始特征数据的最后 seq_length 行
            last_sequence = df[feature_cols].fillna(0).iloc[-seq_length:].values
            
            # 应用与训练数据相同的标准化
            last_sequence = dataset.scaler.transform(last_sequence)
            
            # 转换为张量并添加批次维度
            X_latest_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)
            
            st.write(f"预测数据形状: {X_latest_seq.shape}")  # 调试信息
            
            if X_latest_seq.shape[1] == 0 or X_latest_seq.shape[2] == 0:
                raise ValueError("预测数据维度有误")
                
        except Exception as e:
            st.error(f"准备预测数据时出错: {str(e)}")
            st.error("将尝试备用方法")
            
            # 备用方法：直接使用原始数据创建一个临时预测张量
            temp_features = np.zeros((1, seq_length, len(feature_cols)))
            X_latest_seq = torch.tensor(temp_features, dtype=torch.float32)
        
        # 预测
        prediction_results = {}
        try:
            with torch.no_grad():
                transformer_pred = transformer_model(X_latest_seq).item()
                prediction_results['transformer_prediction'] = transformer_pred
        except Exception as e:
            st.error(f"模型预测出错: {str(e)}")
            # 如果预测失败，使用合理的默认值
            transformer_pred = 0.0
            prediction_results['transformer_prediction'] = transformer_pred
            st.warning("使用默认预测值 0.0")
            
        if ensemble and rf_model is not None:
            # 获取最新的特征向量 (已经过标准化)
            X_latest_features = dataset.features[-1]
            
            try:
                # 获取最新的特征向量
                if dataset.features.shape[0] > 0:
                    X_latest_features = dataset.features[-1]
                else:
                    # 如果特征为空，使用零向量
                    X_latest_features = np.zeros(len(feature_cols))
                
                # 预测
                prediction_results = ensemble_predict(
                    transformer_model, 
                    rf_model, 
                    X_latest_seq, 
                    X_latest_features, 
                    model_weights=[0.6, 0.4]
                )
            except Exception as e:
                st.error(f"集成预测出错: {str(e)}")
                # 使用默认预测结果
                prediction_results = {
                    'ensemble_prediction': transformer_pred,
                    'transformer_prediction': transformer_pred,
                    'rf_prediction': 0.0,
                    'model_agreement_confidence': 0.5
                }
            
        # 显示预测结果
        st.subheader("预测结果")
        
        # 创建结果面板
        result_cols = st.columns(2)
        
        with result_cols[0]:
            if ensemble:
                st.metric(
                    label="集成模型预测的 Confident Score", 
                    value=f"{prediction_results['ensemble_prediction']:.4f}"
                )
                
                # 显示个别模型的预测
                st.write("各模型预测:")
                st.write(f"- Transformer: {prediction_results['transformer_prediction']:.4f}")
                st.write(f"- 随机森林: {prediction_results['rf_prediction']:.4f}")
                st.write(f"- 模型一致性: {prediction_results['model_agreement_confidence']:.2f}")
                
                # 使用集成结果作为最终预测
                final_prediction = prediction_results['ensemble_prediction']
            else:
                st.metric(
                    label="Transformer 预测的 Confident Score", 
                    value=f"{prediction_results['transformer_prediction']:.4f}"
                )
                
                # 使用 Transformer 的结果作为最终预测
                final_prediction = prediction_results['transformer_prediction']
        
        with result_cols[1]:
            # 显示解读
            st.subheader("预测解读")
            st.write(interpret_confidence_score(final_prediction))
            
            # 计算预测的置信区间 (简化版)
            prediction_std = 0.2  # 这里使用一个固定值，实际应用中应该基于模型的不确定性
            lower_bound = final_prediction - 1.96 * prediction_std
            upper_bound = final_prediction + 1.96 * prediction_std
            
            st.write(f"95% 置信区间: [{lower_bound:.4f}, {upper_bound:.4f}]")
            
        # 可视化训练历史
        if 'train_loss' in history and len(history['train_loss']) > 0:
            st.subheader("模型训练历史")
            history_df = pd.DataFrame({
                'epoch': list(range(1, len(history['train_loss']) + 1)),
                'train_loss': history['train_loss'],
                'val_loss': history.get('val_loss', [None] * len(history['train_loss']))
            })
            
            loss_fig = px.line(
                history_df, 
                x='epoch', 
                y=['train_loss', 'val_loss'],
                title="训练损失曲线"
            )
            st.plotly_chart(loss_fig, use_container_width=True)
        
        # 清除进度信息
        progress_container.empty()
        
        st.success("预测完成！")

if __name__ == "__main__":
    main()