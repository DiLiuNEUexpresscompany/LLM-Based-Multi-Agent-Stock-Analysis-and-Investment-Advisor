import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
# news_search_tool.py
import http.client
import urllib.parse
import json
from typing import List, Dict
from datetime import datetime
from tools.base_tool import BaseTool
import os
from dotenv import load_dotenv
import asyncio
# 引入 Crawl4AI 相关依赖
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

class NewsSearchTool(BaseTool):
    """
    使用 NewsData.io API 搜索新闻文章，并使用 Crawl4AI 爬虫爬取每个文章链接指向的页面内容（转换为 Markdown）。
    返回的字段包括：title, description, url, published_at, time_ago, source, author, image_url, categories, sentiment,
    以及爬取的 article_content（Markdown 格式）。
    """

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
        return (
            "搜索最新新闻文章（使用 NewsData.io API），参数：query（必填：搜索关键词），limit（可选，默认为10）。\n"
            "返回基本信息以及通过 Crawl4AI 爬取的文章全文（Markdown 格式）。"
        )

    def _format_date(self, date_str: str) -> str:
        """格式化日期字符串为可读格式"""
        try:
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
        """计算发布时间距离当前的时间"""
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

    def _scrape_article_content(self, url: str) -> str:
        """
        使用 Crawl4AI 异步爬虫爬取给定链接页面内容，并转换为 Markdown 格式返回。
        为了在同步环境中调用异步爬虫，这里使用 asyncio.run()。
        """
        try:
            async def crawl():
                run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url, config=run_conf)
                    # 返回爬取页面的 Markdown 格式内容
                    return result.markdown
            return asyncio.run(crawl())
        except Exception as e:
            return f"Error fetching content with Crawl4AI: {str(e)}"

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
            }, quote_via=urllib.parse.quote)
            
            conn = http.client.HTTPSConnection(self.base_host, timeout=10)
            try:
                request_url = f'{self.base_path}?{params}'
                print(f"Making request to: {self.base_host}{request_url}")
                
                conn.request('GET', request_url)
                response = conn.getresponse()
                print(f"Response status: {response.status}")
                
                if response.status != 200:
                    error_message = response.read().decode('utf-8')
                    print(f"Error response: {error_message}")
                    return [{"error": f"API request failed with status {response.status}", "details": error_message}]
                
                data = response.read()
                raw_response = data.decode('utf-8')
                print(f"Raw API Response: {raw_response[:500]}...")
                
                results_json = json.loads(raw_response)
                print(f"API Status: {results_json.get('status')}")
                
                if results_json.get('status') != 'success':
                    error_msg = results_json.get('message', 'API request failed')
                    print(f"API Error: {error_msg}")
                    return [{"error": error_msg}]
                
                articles = results_json.get('results', [])
                print(f"Found {len(articles)} articles")
                
                if not articles:
                    print("No articles found in response")
                    return [{"error": "No results found for the given query"}]

                formatted_results = []
                limit = int(limit)
                for i, article in enumerate(articles[:limit]):
                    try:
                        pub_date = article.get('pubDate') or ''
                        description = article.get('description') or 'No description available'
                        link = article.get('link') or ''
                        creator_list = article.get('creator') or []
                        author = creator_list[0] if isinstance(creator_list, list) and creator_list else "Unknown"
                        
                        formatted_article = {
                            'title': article.get('title') or 'No title',
                            'description': description[:200] + ('...' if len(description) > 200 else ''),
                            'url': link,
                            'published_at': self._format_date(pub_date) if pub_date else 'Unknown date',
                            'time_ago': self._calculate_time_ago(pub_date) if pub_date else 'Unknown time',
                            'source': article.get('source_id') or 'Unknown source',
                            'author': author,
                            'image_url': article.get('image_url') or '',
                            'categories': article.get('category') or [],
                            'sentiment': article.get('sentiment') or ''
                        }
                        
                        # 使用 Crawl4AI 爬取每个新闻链接的文章内容
                        if link:
                            print(f"Scraping content from link: {link}")
                            article_content = self._scrape_article_content(link)
                            formatted_article['article_content'] = article_content
                        else:
                            formatted_article['article_content'] = 'No link provided'
                        
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
