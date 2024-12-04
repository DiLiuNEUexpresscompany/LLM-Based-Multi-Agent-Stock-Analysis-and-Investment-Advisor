import http.client
import json
from typing import List, Dict
from datetime import datetime,timedelta
from tools.base_tool import BaseTool
import os
from dotenv import load_dotenv

class StockPriceTool(BaseTool):
    """Tool for searching stock price using PLOYGON.io API"""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("STOCK_API_KEY")
        if not self.api_key:
            raise ValueError("Stock_API_KEY not found in environment variables")
        self.base_host = 'api.polygon.io'
        self.base_path = '/v1/open-close'
    def name(self) -> str:
        return "fetch_stock_data"
    def description(self) -> str:
        return """Fetch historical stock data for a given ticker and date range.
        Arguments:
        - ticker (required): Stock ticker symbol (e.g., 'AAPL')
        - days (required): Number of past days to fetch data for"""
    def _format_date(self, date_obj: datetime) -> str:
        """Format date as YYYY-MM-DD"""
        return date_obj.strftime('%Y-%m-%d')
    def _process_stock_data(self, data: Dict, date: str) -> Dict:
        """Process and validate the stock data response"""
        if data.get('status') != 'OK':
            return {
                'date': date,
                'error': data.get('message', 'Unknown error occurred')
            }
            
        return {
            'date': data.get('from', date),
            'symbol': data.get('symbol'),
            'prices': {
                'open': round(data.get('open', 0), 2),
                'high': round(data.get('high', 0), 2),
                'low': round(data.get('low', 0), 2),
                'close': round(data.get('close', 0), 2),
            },
            'trading_data': {
                'volume': data.get('volume', 0),
                'pre_market': round(data.get('preMarket', 0), 2),
                'after_hours': round(data.get('afterHours', 0), 2)
            },
            'status': data.get('status')
        }
    def execute(self, ticker: str = None, days: int = 1, date: str = None, **kwargs) -> List[Dict]:
        print(f"Executing stock data fetch for ticker: {ticker}, date: {date}")
        
        if not ticker or not isinstance(ticker, str):
            print("Invalid ticker parameter")
            return [{"error": "A valid stock ticker is required"}]
        
        try:
            # 如果提供了具体日期，直接使用该日期
            if date:
                try:
                    path = f"{self.base_path}/{ticker}/{date}?apiKey={self.api_key}"
                    
                    conn = http.client.HTTPSConnection(self.base_host, timeout=10)
                    conn.request('GET', path)
                    response = conn.getresponse()
                    
                    if response.status != 200:
                        error_message = response.read().decode('utf-8')
                        print(f"Error response for {date}: {error_message}")
                        return [{
                            "date": date,
                            "error": f"API request failed with status {response.status}"
                        }]
                    
                    data = response.read()
                    stock_data = json.loads(data.decode('utf-8'))
                    
                    # Process the response data
                    processed_data = self._process_stock_data(stock_data, date)
                    return [processed_data]
                    
                except Exception as e:
                    print(f"Error processing {date}: {str(e)}")
                    return [{
                        "date": date,
                        "error": f"Failed to process data: {str(e)}"
                    }]
            
            # 如果没有提供具体日期，使用原来的日期范围逻辑
            else:
                results = []
                end_date = datetime.now()
                
                for i in range(days):
                    current_date = end_date - timedelta(days=i)
                    formatted_date = self._format_date(current_date)
                    
                    try:
                        path = f"{self.base_path}/{ticker}/{formatted_date}?apiKey={self.api_key}"
                        
                        conn = http.client.HTTPSConnection(self.base_host, timeout=10)
                        conn.request('GET', path)
                        response = conn.getresponse()
                        
                        if response.status != 200:
                            error_message = response.read().decode('utf-8')
                            print(f"Error response for {formatted_date}: {error_message}")
                            results.append({
                                "date": formatted_date,
                                "error": f"API request failed with status {response.status}"
                            })
                            continue
                        
                        data = response.read()
                        stock_data = json.loads(data.decode('utf-8'))
                        
                        # Process the response data
                        processed_data = self._process_stock_data(stock_data, formatted_date)
                        results.append(processed_data)
                        
                        print(f"Successfully retrieved data for {formatted_date}")
                        
                    except Exception as e:
                        print(f"Error processing {formatted_date}: {str(e)}")
                        results.append({
                            "date": formatted_date,
                            "error": f"Failed to process data: {str(e)}"
                        })
                    finally:
                        conn.close()
                
                if not results:
                    return [{"error": "No data was successfully retrieved"}]
                
                return results

        except Exception as e:
            print(f"Unexpected error in execute: {str(e)}")
            return [{"error": f"Unexpected error: {str(e)}"}]
            