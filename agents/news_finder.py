# news_agent.py
from typing import Dict, List
from .base_agent import BaseAgent
import json

class NewsFinder(BaseAgent):
    def __init__(self, registry):
        super().__init__(registry)
        self.role = "News Finder"
        self.goal = "Find and retrieve relevant news articles"
        self.backstory = "An agent specialized in locating the latest news stories."
        self.tools.append("news_search")
    def format_tool_result(self, result: List[Dict]) -> str:
        """Format tool results for the LLM to process"""
        try:
            # 如果结果是错误信息
            if isinstance(result, list) and result and "error" in result[0]:
                return json.dumps({
                    "status": "error",
                    "message": result[0].get("error", "Unknown error")
                })
            
            # 如果是成功的新闻结果
            if isinstance(result, list):
                formatted_news = {
                    "status": "success",
                    "count": len(result),
                    "articles": []
                }
                
                for article in result:
                    formatted_article = {
                        "title": article.get("title", ""),
                        "source": article.get("source", ""),
                        "published_at": article.get("published_at", ""),
                        "description": article.get("description", "")[:200] if article.get("description") else "",
                        "url": article.get("url", "")
                    }
                    formatted_news["articles"].append(formatted_article)
                
                return json.dumps(formatted_news, ensure_ascii=False, indent=2)
            
            return json.dumps({"status": "unknown", "data": result})
            
        except Exception as e:
            print(f"Error in format_tool_result: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Error formatting results: {str(e)}"
            })

    def get_system_prompt(self, system_prompt = None) -> str:
        if system_prompt is not None:
            return system_prompt
        
        tools_info = self.registry.get_tools_info()
        return f"""
            You are a news search assistant. You help find and analyze news articles.
            When you receive search results, please:
            1. For successful searches, provide a brief summary of the most relevant articles found, 
            including their titles, sources, and key points.
            2. If you find multiple related articles, group them by topic.
            3. Always include the most important details first.
            4. If there are any errors, explain them clearly and suggest alternatives.

            Available tools:
            {tools_info}
            
            Call tools using format:
            <tool_call>{{"name": "search_news", "arguments": {{"query": "search terms"}}}}</tool_call>
            """
    def process_tool_arguments(self, tool_name: str, arguments: Dict) -> Dict:
        if tool_name == "search_news":
            if "query" in arguments and arguments["query"]:
                return arguments
            if "keyword" in arguments and arguments["keyword"]:
                arguments["query"] = arguments.pop("keyword")
                return arguments
            raise ValueError("A valid search query is required")
        return arguments