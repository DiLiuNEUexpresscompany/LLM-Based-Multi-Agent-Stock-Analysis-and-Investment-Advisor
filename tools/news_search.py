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
        self.default_limit = 5

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

    def execute(self, query: str = None, limit: int = 5, **kwargs) -> List[Dict]:
        # Parameter validation
        if not query or not isinstance(query, str):
            return [{"error": "A valid search query is required"}]
        
        try:
            limit = int(limit)
            if limit < 1 or limit > 50:
                limit = self.default_limit
        except (TypeError, ValueError):
            limit = self.default_limit

        try:
            # Build API request
            params = urllib.parse.urlencode({
                'apikey': self.api_key,
                'qInTitle': query,
                'language': 'en'
            })

            # Create connection with timeout
            conn = http.client.HTTPSConnection(self.base_host, timeout=10)
            
            try:
                # Make request
                conn.request('GET', f'{self.base_path}?{params}')
                response = conn.getresponse()
                
                # Handle response
                if response.status != 200:
                    error_message = response.read().decode('utf-8')
                    return [{"error": f"API request failed with status {response.status}", "details": error_message}]
                
                # Parse response
                data = response.read()
                results = json.loads(data.decode('utf-8'))
                
                # Validate API response
                if results.get('status') != 'success':
                    return [{"error": results.get('message', 'API request failed')}] 

                # Format results for agent
                formatted_results = []
                for article in results.get('results', [])[:limit]:
                    pub_date = article.get('pubDate', '')
                    description = article.get('description', 'No description available')
                    
                    formatted_results.append({
                        'title': article.get('title', 'No title'),
                        'description': description[:200] + ('...' if len(description) > 200 else ''),
                        'url': article.get('link', ''),
                        'published_at': self._format_date(pub_date),
                        'time_ago': self._calculate_time_ago(pub_date),
                        'source': article.get('source_id', 'Unknown source'),
                        'author': article.get('creator', ['Unknown'])[0] if article.get('creator') else 'Unknown',
                        'image_url': article.get('image_url', ''),
                        'categories': article.get('category', []),
                    })

                if not formatted_results:
                    return [{"error": "No results found for the given query"}]
                
                return formatted_results

            finally:
                conn.close()

        except json.JSONDecodeError as e:
            return [{"error": f"Failed to parse API response: {str(e)}"}]
        except http.client.HTTPException as e:
            return [{"error": f"HTTP request failed: {str(e)}"}]
        except TimeoutError:
            return [{"error": "Request timed out"}]
        except Exception as e:
            return [{"error": f"Unexpected error: {str(e)}"}]
