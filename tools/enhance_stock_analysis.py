from typing import Dict, List, Union, Optional
from tools.base_tool import BaseTool
from datetime import datetime
import numpy as np
from dataclasses import dataclass

@dataclass
class StockMetrics:
    """Data class for storing stock analysis metrics"""
    current_price: float
    price_change: float
    price_change_percent: float
    volatility: float
    moving_averages: Dict[str, float]
    rsi: float
    beta: Optional[float] = None
    sharpe_ratio: Optional[float] = None

class StockAnalysisTool(BaseTool):
    """Enhanced tool for mathematical analysis of stock data"""
    
    def name(self) -> str:
        return "analyze_stock_data"
        
    def description(self) -> str:
        return """
        Perform mathematical analysis on stock data to calculate various technical and statistical metrics.
        
        Arguments:
        - stock_data (required): List of daily stock data dictionaries
        - market_data (optional): List of market index data for beta calculation
        - risk_free_rate (optional): Risk-free rate for Sharpe ratio calculation
        - metrics (optional): List of specific metrics to calculate
        - ma_periods (optional): List of periods for moving average calculation
        """
    
    def _validate_data(self, data: List[Dict]) -> List[float]:
        """Validate and extract closing prices from stock data"""
        if not data or not isinstance(data, list):
            raise ValueError("Invalid stock data format")
        
        prices = []
        for day in data:
            if not isinstance(day, dict) or 'prices' not in day:
                continue
            try:
                prices.append(float(day['prices']['close']))
            except (KeyError, ValueError, TypeError):
                continue
                
        if not prices:
            raise ValueError("No valid price data found")
        
        return prices
    
    def _calculate_returns(self, prices: List[float]) -> np.ndarray:
        """Calculate logarithmic returns for more accurate statistical analysis"""
        prices = np.array(prices)
        returns = np.log(prices[1:] / prices[:-1])
        return returns
    
    def _calculate_volatility(self, returns: np.ndarray, annualize: bool = True) -> float:
        """Calculate volatility using standard deviation of log returns"""
        if len(returns) < 2:
            return 0.0
        vol = np.std(returns, ddof=1)
        return vol * np.sqrt(252) if annualize else vol
    
    def _calculate_moving_averages(self, prices: List[float], 
                                 periods: List[int] = None) -> Dict[str, float]:
        """Calculate exponential moving averages for specified periods"""
        if not periods:
            periods = [5, 10, 20, 50]
            
        ma_dict = {}
        prices_arr = np.array(prices)
        
        for period in periods:
            if len(prices) >= period:
                weights = np.exp(np.linspace(-1., 0., period))
                weights /= weights.sum()
                ma = np.convolve(prices_arr, weights, mode='valid')[0]
                ma_dict[f'{period}_day'] = float(ma)
            else:
                ma_dict[f'{period}_day'] = None
                
        return ma_dict
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI using exponential moving averages"""
        if len(prices) < period + 1:
            return 50.0
            
        returns = np.diff(prices)
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            return 100.0
            
        for i in range(period, len(returns)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        return 100.0 - (100.0 / (1.0 + rs))
    
    def _calculate_beta(self, stock_returns: np.ndarray, 
                       market_returns: np.ndarray) -> Optional[float]:
        """Calculate beta coefficient against market returns"""
        if len(stock_returns) != len(market_returns) or len(stock_returns) < 2:
            return None
            
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns, ddof=1)
        
        return covariance / market_variance if market_variance != 0 else None
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, 
                              risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio using returns and risk-free rate"""
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns, ddof=1)
    
    def execute(self, stock_data: List[Dict], market_data: List[Dict] = None,
               risk_free_rate: float = 0.02, metrics: List[str] = None,
               ma_periods: List[int] = None) -> Dict:
        """
        Execute stock analysis with specified parameters and metrics
        """
        try:
            # Validate and prepare price data
            prices = self._validate_data(stock_data)
            returns = self._calculate_returns(prices)
            
            # Calculate basic metrics
            metrics = StockMetrics(
                current_price=prices[0],
                price_change=prices[0] - prices[-1],
                price_change_percent=((prices[0] - prices[-1]) / prices[-1] * 100),
                volatility=self._calculate_volatility(returns),
                moving_averages=self._calculate_moving_averages(prices, ma_periods),
                rsi=self._calculate_rsi(prices)
            )
            
            # Calculate additional metrics if market data is provided
            if market_data:
                market_prices = self._validate_data(market_data)
                market_returns = self._calculate_returns(market_prices)
                metrics.beta = self._calculate_beta(returns, market_returns)
                metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns, risk_free_rate)
            
            # Prepare analysis result
            analysis = {
                "symbol": stock_data[0].get('symbol', 'Unknown'),
                "analysis_date": datetime.now().strftime('%Y-%m-%d'),
                "period": {
                    "start": stock_data[-1].get('date'),
                    "end": stock_data[0].get('date')
                },
                "metrics": {
                    "current_price": round(metrics.current_price, 2),
                    "price_change": round(metrics.price_change, 2),
                    "price_change_percent": round(metrics.price_change_percent, 2),
                    "volatility": round(metrics.volatility * 100, 2),  # Convert to percentage
                    "moving_averages": {k: round(v, 2) if v is not None else None 
                                      for k, v in metrics.moving_averages.items()},
                    "rsi": round(metrics.rsi, 2)
                },
                "advanced_metrics": {
                    "beta": round(metrics.beta, 2) if metrics.beta is not None else None,
                    "sharpe_ratio": round(metrics.sharpe_ratio, 2) if metrics.sharpe_ratio is not None else None
                }
            }
            
            # Generate trading signals
            analysis["signals"] = {
                "rsi_signal": "oversold" if metrics.rsi < 30 else 
                             "overbought" if metrics.rsi > 70 else "neutral",
                "trend": "uptrend" if any(v > metrics.current_price 
                                        for v in metrics.moving_averages.values() if v is not None)
                        else "downtrend",
                "volatility_level": "high" if metrics.volatility > 0.02 else 
                                  "medium" if metrics.volatility > 0.01 else "low"
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}