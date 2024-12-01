from typing import Dict, List
from .base_agent import BaseAgent
import json
import re
from datetime import datetime

class StockPriceAgent(BaseAgent):
    def format_tool_result(self, result: List[Dict]) -> str:
        """Format tool results for the LLM to process"""
        try:
            # Handle error results
            if isinstance(result, list) and result and "error" in result[0]:
                return json.dumps({
                    "status": "error",
                    "message": result[0].get("error", "Unknown error")
                })
            
            # Handle successful stock data results
            if isinstance(result, list):
                formatted_data = {
                    "status": "success",
                    "count": len(result),
                    "stock_data": []
                }
                
                for day_data in result:
                    if "error" in day_data:
                        # Skip days with errors
                        continue
                        
                    formatted_day = {
                        "date": day_data.get("date", ""),
                        "symbol": day_data.get("symbol", ""),
                        "prices": day_data.get("prices", {}),
                        "trading_data": {
                            "volume": day_data.get("trading_data", {}).get("volume", 0),
                            "pre_market": day_data.get("trading_data", {}).get("pre_market", 0),
                            "after_hours": day_data.get("trading_data", {}).get("after_hours", 0)
                        }
                    }
                    formatted_data["stock_data"].append(formatted_day)
                
                return json.dumps(formatted_data, ensure_ascii=False, indent=2)
            
            return json.dumps({"status": "unknown", "data": result})
            
        except Exception as e:
            print(f"Error in format_tool_result: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Error formatting results: {str(e)}"
            })

    def get_system_prompt(self) -> str:
        tools_info = self.registry.get_tools_info()
        return f"""
        You are a stock price analysis assistant. You help fetch and analyze stock price data.
        When you receive stock data, please:
        1. For successful queries, provide a concise summary of the stock's performance,
           including key metrics like opening/closing prices, highs/lows, and trading volume.
        2. Calculate and mention any significant price changes or trends across the requested period.
        3. Highlight any notable pre-market or after-hours price movements.
        4. If there are any errors, explain them clearly and suggest alternatives.

        When user asks for stock data:
        - If a specific date is provided (e.g., "2024-11-29"), use that date
        - If no date is provided, use the most recent trading day

        Available tools:
        {tools_info}
        
        Call tools using format:
        <tool_call>{{"name": "fetch_stock_data", "arguments": {{"ticker": "NVDA", "date": "2024-11-29"}}}}</tool_call>
        """

    def extract_date_from_query(self, query: str) -> str:
        """从查询中提取日期"""
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', query)
        if date_match:
            return date_match.group(0)
        return None

    def extract_ticker_from_query(self, query: str) -> str:
        """从查询中提取股票代码"""
        ticker_match = re.search(r'\b[A-Z]+\b', query)
        if ticker_match:
            return ticker_match.group(0)
        return None

    def process_tool_arguments(self, tool_name: str, arguments: Dict) -> Dict:
        if tool_name == "fetch_stock_data":
            # 处理股票代码
            if "ticker" not in arguments or not arguments["ticker"]:
                ticker = self.extract_ticker_from_query(self.current_query)
                if ticker:
                    arguments["ticker"] = ticker.upper()
                else:
                    raise ValueError("A valid stock ticker is required")
            else:
                arguments["ticker"] = arguments["ticker"].upper()
            
            # 处理日期
            if "date" not in arguments:
                date = self.extract_date_from_query(self.current_query)
                if date:
                    arguments["date"] = date
            
            return arguments
        return arguments

    def run(self, query: str) -> str:
        """重写run方法来保存当前查询"""
        self.current_query = query  # 保存当前查询以供提取参数使用
        return super().run(query)