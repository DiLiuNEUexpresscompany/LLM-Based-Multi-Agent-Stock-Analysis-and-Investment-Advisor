from typing import Dict, List
from .base_agent import BaseAgent
from datetime import datetime, timedelta
import logging
import numpy as np
import json
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

class PriceTracker(BaseAgent):
    """Enhanced agent for comprehensive daily stock analysis"""
    
    def __init__(self, registry: 'ToolRegistry'):
        super().__init__(registry)
        self.last_day_data = None  # Stores the most recent stock data for analysis.
        self.role = "Price Tracker"  # Defines the role of the module.
        self.goal = (
            "To analyze recent stock price data for a given company, focusing on short-term price movements, "
            "key support and resistance levels, volume trends, and identifying potential patterns such as breakouts "
            "or pullbacks. The analysis should provide actionable insights into potential trading opportunities."
        )
        self.backstory = (
            "This module is designed to assist users by providing detailed and visually supported stock market analyses. "
            "Its primary purpose is to highlight significant trends and movements in the stock price, empowering users to make "
            "informed trading decisions."
        )
        self.tools = [
            ["stock_price", "Used to retrieve and analyze stock price data."],
            ["data_analysis", "Used for trend analysis, visualization, and identifying trading patterns."]
        ]
    def get_system_prompt(self, system_prompt = None) -> str:
        if system_prompt is not None:
            return system_prompt
        
        tools_desc = "\n".join([
            f"- {tool.name()}: {tool.description()}" 
            for tool in self.registry.get_all_tools()
        ])
        
        return f"""You are an advanced daily stock analysis assistant that provides comprehensive trading metrics and analysis.

        Available tools:
        {tools_desc}
        
        Analyze and explain:
        1. Intraday volatility and price movements
        2. Volume analysis and turnover metrics
        3. Extended hours trading patterns
        4. Technical indicators and market dynamics
        5. Trading volume and liquidity analysis
        
        Focus on:
        - Price movements and volatility metrics
        - Volume and turnover analysis
        - Extended hours trading patterns
        - Comparative price and volume analysis
        - Market microstructure insights
        
        Wrap tool calls in <tool_call> tags using JSON format:
        <tool_call>{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}</tool_call>
        """

    def process_tool_arguments(self, tool_name: str, arguments: Dict) -> Dict:
        """Process and validate tool arguments for daily analysis"""
        try:
            if tool_name == "fetch_stock_data":
                # Validate ticker
                arguments["ticker"] = str(arguments.get("ticker", "")).upper()
                if not arguments["ticker"]:
                    raise ValueError("Stock ticker is required")
                
                # Set date to yesterday
                arguments["date"] = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                logger.info(f"Fetching data for {arguments['ticker']} on {arguments['date']}")
                
            elif tool_name == "analyze_stock_data":
                if "stock_data" not in arguments:
                    raise ValueError("Stock data is required for analysis")
                
                # Store data for volume comparison
                self.last_day_data = arguments.get("stock_data")
                
                # Add default periods if not provided
                if "ma_periods" not in arguments:
                    arguments["ma_periods"] = [5, 10, 20]
                
                # Add default risk-free rate if not provided
                if "risk_free_rate" not in arguments:
                    arguments["risk_free_rate"] = 0.02
                    
                logger.info(f"Analyzing stock data with parameters: {arguments}")
            
            return arguments
            
        except Exception as e:
            logger.error(f"Error processing tool arguments: {str(e)}")
            raise

    def format_tool_result(self, result: Dict) -> str:
        """Format enhanced daily analysis results"""
        try:
            if isinstance(result, list) and result and isinstance(result[0], dict):
                if "error" in result[0]:
                    return str(result[0]["error"])
                return self._format_daily_result(result[0])
                
            if isinstance(result, dict):
                if "error" in result:
                    return str(result["error"])
                return self._format_daily_result(result)
                
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"Error formatting tool result: {str(e)}")
            return f"Error formatting result: {str(e)}"

    def _calculate_metrics(self, data: Dict) -> Dict:
        """Calculate comprehensive trading metrics"""
        try:
            prices = data.get('prices', {})
            trading_data = data.get('trading_data', {})
            
            open_price = float(prices.get('open', 0))
            close_price = float(prices.get('close', 0))
            high_price = float(prices.get('high', 0))
            low_price = float(prices.get('low', 0))
            volume = int(trading_data.get('volume', 0))
            pre_market = float(trading_data.get('pre_market', 0))
            after_hours = float(trading_data.get('after_hours', 0))

            # Price metrics
            day_change = close_price - open_price
            day_change_pct = (day_change / open_price * 100) if open_price else 0
            price_range = high_price - low_price
            price_range_pct = (price_range / low_price * 100) if low_price else 0
            
            # Volatility metrics
            typical_price = (high_price + low_price + close_price) / 3
            prices_array = np.array([open_price, high_price, low_price, close_price])
            price_std = np.std(prices_array)
            intraday_volatility = (price_std / typical_price * 100) if typical_price else 0
            
            # Volume metrics
            value_traded = volume * typical_price if typical_price else 0
            avg_trade_value = value_traded / volume if volume else 0
            
            # Extended hours analysis
            pre_market_change = (pre_market - close_price) / close_price * 100 if close_price and pre_market else 0
            after_hours_change = (after_hours - close_price) / close_price * 100 if close_price and after_hours else 0
            
            return {
                "price_metrics": {
                    "open": open_price,
                    "close": close_price,
                    "high": high_price,
                    "low": low_price,
                    "day_change": day_change,
                    "day_change_pct": day_change_pct,
                    "price_range": price_range,
                    "price_range_pct": price_range_pct,
                    "typical_price": typical_price
                },
                "volatility_metrics": {
                    "intraday_volatility": intraday_volatility,
                    "price_std": price_std,
                    "high_low_range": price_range_pct
                },
                "volume_metrics": {
                    "volume": volume,
                    "value_traded": value_traded,
                    "avg_trade_value": avg_trade_value
                },
                "extended_hours": {
                    "pre_market": pre_market,
                    "after_hours": after_hours,
                    "pre_market_change": pre_market_change,
                    "after_hours_change": after_hours_change
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def _format_daily_result(self, data: Dict) -> str:
        """Format a single day's trading data with enhanced metrics"""
        try:
            metrics = self._calculate_metrics(data)
            if not metrics:
                return "Unable to calculate trading metrics"
            
            formatted = (
                f"Enhanced Daily Analysis for {data.get('symbol', 'Unknown')} - {data.get('date')}\n\n"
                
                f"Price Performance:\n"
                f"• Opening: ${metrics['price_metrics']['open']:.2f}\n"
                f"• Closing: ${metrics['price_metrics']['close']:.2f}\n"
                f"• High/Low: ${metrics['price_metrics']['high']:.2f} / ${metrics['price_metrics']['low']:.2f}\n"
                f"• Price Change: ${metrics['price_metrics']['day_change']:.2f} ({metrics['price_metrics']['day_change_pct']:.2f}%)\n"
                f"• Trading Range: ${metrics['price_metrics']['price_range']:.2f} ({metrics['price_metrics']['price_range_pct']:.2f}%)\n\n"
                
                f"Volatility Metrics:\n"
                f"• Intraday Volatility: {metrics['volatility_metrics']['intraday_volatility']:.2f}%\n"
                f"• Price Std Dev: ${metrics['volatility_metrics']['price_std']:.2f}\n"
                f"• High-Low Range: {metrics['volatility_metrics']['high_low_range']:.2f}%\n\n"
                
                f"Volume Analysis:\n"
                f"• Total Volume: {metrics['volume_metrics']['volume']:,} shares\n"
                f"• Value Traded: ${metrics['volume_metrics']['value_traded']:,.2f}\n"
                f"• Avg Trade Value: ${metrics['volume_metrics']['avg_trade_value']:.2f}\n\n"
                
                f"Extended Hours Activity:\n"
                f"• Pre-Market: ${metrics['extended_hours']['pre_market']:.2f} ({metrics['extended_hours']['pre_market_change']:.2f}%)\n"
                f"• After-Hours: ${metrics['extended_hours']['after_hours']:.2f} ({metrics['extended_hours']['after_hours_change']:.2f}%)\n"
            )
            
            insights = self._generate_enhanced_insights(metrics)
            if insights:
                formatted += "\nKey Insights:\n" + "\n".join(f"• {insight}" for insight in insights)
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting daily result: {str(e)}")
            return f"Error formatting daily result: {str(e)}"

    def _generate_enhanced_insights(self, metrics: Dict) -> List[str]:
        """Generate comprehensive trading insights"""
        insights = []
        try:
            price = metrics['price_metrics']
            vol = metrics['volatility_metrics']
            volume = metrics['volume_metrics']
            ext = metrics['extended_hours']
            
            # Price movement insights
            if abs(price['day_change_pct']) > 2:
                direction = "gained" if price['day_change_pct'] > 0 else "declined"
                insights.append(
                    f"Stock {direction} significantly by {abs(price['day_change_pct']):.1f}%"
                )
            
            # Volatility insights
            if vol['intraday_volatility'] > 2:
                insights.append(
                    f"High intraday volatility detected at {vol['intraday_volatility']:.1f}%"
                )
            
            # Volume insights
            if volume['volume'] > 1000000:
                insights.append(
                    f"Heavy trading volume with {volume['volume']:,} shares traded"
                )
            
            # Extended hours insights
            if abs(ext['pre_market_change']) > 1:
                insights.append(
                    f"Significant pre-market activity ({ext['pre_market_change']:.1f}% change)"
                )
            if abs(ext['after_hours_change']) > 1:
                insights.append(
                    f"Notable after-hours movement ({ext['after_hours_change']:.1f}% change)"
                )
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return ["Error generating insights"]