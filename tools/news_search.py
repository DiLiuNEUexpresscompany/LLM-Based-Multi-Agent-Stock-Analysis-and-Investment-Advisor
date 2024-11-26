import http.client
import urllib.parse
import json
from typing import List, Dict
from datetime import datetime
from tools.base_tool import BaseTool
import os
from dotenv import load_dotenv

class NewsSearchTool(BaseTool):
    """Tool for searching news articles using TheNewsAPI"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.api_token = os.getenv("THENEWS_API_KEY")
        if not self.api_token:
            raise ValueError("THENEWS_API_KEY not found in environment variables")
        
        self.base_host = 'api.thenewsapi.com'
        self.base_path = '/v1/news/all'

    def name(self) -> str:
        return "search_news"
    
    def description(self) -> str:
        return "Search recent news articles by keyword. Arguments: query (required), limit (optional, default=5), categories (optional, default='tech,business')"

    def _format_date(self, date_str: str) -> str:
        """Converts ISO format to readable date."""
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return date_str

    def _time_since_published(self, published_at: str) -> str:
        """Calculates the time since the article was published."""
        try:
            dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            delta = datetime.utcnow() - dt
            days = delta.days
            hours = delta.seconds // 3600
            if days > 0:
                return f"{days} days ago"
            return f"{hours} hours ago"
        except Exception:
            return "Unknown time"

    def _validate_inputs(self, query: str, limit: int, categories: str) -> Dict:
        """Validate input parameters"""
        if not query.strip():
            return {"error": "Search query cannot be empty"}
        if limit < 1:
            return {"error": "Limit must be greater than 0"}
        if limit > 50:
            return {"error": "Limit cannot exceed 50"}
        return {}

    def execute(self, query: str, limit: int = 10, categories: str = "tech,business") -> List[Dict]:
        # Validate inputs
        validation_error = self._validate_inputs(query, limit, categories)
        if validation_error:
            return [validation_error]

        try:
            # Build request parameters
            params = urllib.parse.urlencode({
                'api_token': self.api_token,
                'search': query,
                'categories': categories,
                'limit': limit,
                'language': 'en',
                'sort': 'published_at'
            })

            # Create connection with timeout
            conn = http.client.HTTPSConnection(self.base_host, timeout=10)
            
            try:
                # Send request
                conn.request('GET', f'{self.base_path}?{params}')
                response = conn.getresponse()
                
                # Check response status
                if response.status != 200:
                    error_message = response.read().decode('utf-8')
                    return [{
                        "error": f"API request failed with status {response.status}",
                        "details": error_message
                    }]
                
                # Parse response data
                data = response.read()
                results = json.loads(data.decode('utf-8'))
                
                # Check API return for errors
                if 'error' in results:
                    return [{"error": results['error']}]
                
                # Format the results
                formatted_results = []
                for article in results.get('data', [])[:limit]:
                    formatted_results.append({
                        'title': article.get('title', 'No title'),
                        'description': article.get('description', 'No description available'),
                        'url': article.get('url', ''),
                        'image_url': article.get('image_url', ''),
                        'published_at': self._time_since_published(article.get('published_at', '')),
                        'source': article.get('source', 'Unknown'),
                        'relevance_score': article.get('relevance_score', None)  # Include relevance_score
                    })
                
                return formatted_results or [{"error": "No results found"}]

            finally:
                # Ensure connection is closed
                conn.close()

        except json.JSONDecodeError as e:
            return [{"error": f"Failed to parse API response: {str(e)}"}]
        except http.client.HTTPException as e:
            return [{"error": f"HTTP request failed: {str(e)}"}]
        except TimeoutError:
            return [{"error": "Request timed out"}]
        except Exception as e:
            return [{"error": f"Unexpected error: {str(e)}"}]
