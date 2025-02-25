#news_search_tool.py
import http.client
import urllib.parse
import json
from typing import List, Dict
from datetime import datetime
from tools.base_tool import BaseTool
import os
from dotenv import load_dotenv


class NewsSearchTool(BaseTool):
    """Tool for searching news articles using NewsData.io API"""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("NEWS_API_KEY not found in environment variables")
        
        self.base_host = 'newsdata.io'
        self.base_path = '/api/1/latest'
        self.default_limit = 10

    def name(self) -> str:
        return "search_news"
    
    def description(self) -> str:
        return """Search recent news articles by keyword. 
        Arguments: 
        - query (required): Search keyword or phrase
        - limit (optional, default=5): Number of articles to return"""

    def _format_date(self, date_str: str) -> str:
        """Format date string to readable format"""
        try:
            # Handle multiple possible date formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"]:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
            return date_str
        except Exception:
            return date_str

    def _calculate_time_ago(self, pub_date: str) -> str:
        """Calculate time since publication"""
        try:
            formatted_date = self._format_date(pub_date)
            dt = datetime.strptime(formatted_date, '%Y-%m-%d %H:%M:%S')
            now = datetime.now()
            diff = now - dt
            
            if diff.days > 0:
                return f"{diff.days} days ago"
            hours = diff.seconds // 3600
            if hours > 0:
                return f"{hours} hours ago"
            minutes = (diff.seconds % 3600) // 60
            return f"{minutes} minutes ago"
        except Exception:
            return "time unknown"

    def execute(self, query: str = None, limit: int = 10, **kwargs) -> List[Dict]:
        print(f"Executing search with query: {query}")
        
        if not query or not isinstance(query, str):
            print("Invalid query parameter")
            return [{"error": "A valid search query is required"}]
        
        try:
            params = urllib.parse.urlencode({
                'apikey': self.api_key,
                'qIntitle': query,
                'language': 'en'
            })
            
            conn = http.client.HTTPSConnection(self.base_host, timeout=10)
            try:
                # 打印完整请求信息
                print(f"Making request to: {self.base_host}{self.base_path}?{params}")
                
                conn.request('GET', f'{self.base_path}?{params}')
                response = conn.getresponse()
                print(f"Response status: {response.status}")
                
                if response.status != 200:
                    error_message = response.read().decode('utf-8')
                    print(f"Error response: {error_message}")
                    return [{"error": f"API request failed with status {response.status}", "details": error_message}]
                
                # 读取并打印原始响应数据
                data = response.read()
                raw_response = data.decode('utf-8')
                print(f"Raw API Response: {raw_response[:500]}...")  # 只打印前500个字符
                
                results = json.loads(raw_response)
                print(f"API Status: {results.get('status')}")
                
                if results.get('status') != 'success':
                    error_msg = results.get('message', 'API request failed')
                    print(f"API Error: {error_msg}")
                    return [{"error": error_msg}]

                articles = results.get('results', [])
                print(f"Found {len(articles)} articles")
                
                if not articles:
                    print("No articles found in response")
                    return [{"error": "No results found for the given query"}]

                formatted_results = []
                limit = int(limit)
                for i, article in enumerate(articles[:limit]):
                    try:
                        pub_date = article.get('pubDate', '')
                        description = article.get('description', 'No description available')
                        
                        formatted_article = {
                            'title': article.get('title', 'No title'),
                            'description': description[:200] + ('...' if len(description) > 200 else ''),
                            'url': article.get('link', ''),
                            'published_at': self._format_date(pub_date) if pub_date else 'Unknown date',
                            'time_ago': self._calculate_time_ago(pub_date) if pub_date else 'Unknown time',
                            'source': article.get('source_id', 'Unknown source'),
                            'author': article.get('creator', ['Unknown'])[0] if article.get('creator') else 'Unknown',
                            'image_url': article.get('image_url', ''),
                            'categories': article.get('category', [])
                        }
                        print(f"Successfully formatted article {i+1}")
                        formatted_results.append(formatted_article)
                    except Exception as e:
                        print(f"Error formatting article {i+1}: {str(e)}")
                        continue

                if not formatted_results:
                    print("No results were successfully formatted")
                    return [{"error": "Failed to format any news results"}]
                
                print(f"Successfully formatted {len(formatted_results)} articles")
                return formatted_results

            finally:
                conn.close()

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {str(e)}\nResponse data: {raw_response[:500]}")
            return [{"error": f"Failed to parse API response: {str(e)}"}]
        except Exception as e:
            print(f"Unexpected error in execute: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return [{"error": f"Unexpected error: {str(e)}"}]