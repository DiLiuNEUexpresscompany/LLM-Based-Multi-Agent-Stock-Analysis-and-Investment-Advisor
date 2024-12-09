# news_agent.py
from typing import Dict, List
from .base_agent import BaseAgent
import json

class NewsSearcher(BaseAgent):
    def __init__(self, registry):
        super().__init__(registry)
        self.role = "News Searcher"  # Defines the role of the module.
        self.goal = (
            "To research and gather the latest news related to a specified company, focusing on significant events, "
            "market sentiment, analyst opinions, and upcoming developments like product launches or earnings reports. "
            "The aim is to provide actionable insights into how these factors might impact the company's stock performance."
        )
        self.backstory = (
            "This module is built to assist users in staying informed about the latest developments related to specific companies. "
            "By synthesizing news, market sentiment, and expert opinions, it helps users understand the potential implications "
            "of current events on stock performance."
        )
        self.tools = [
            ["news_search", "Tool for retrieving and analyzing the latest news."]
        ]

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
    
    def run(self, user_input: str) -> str:
        system_prompt = self.get_system_prompt()
        # Get tool call from LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        
        tool_call = self._parse_tool_call(response.choices[0].message.content)
        if not tool_call:
            # If no tool call is found, directly use the LLM's response
            return response.choices[0].message.content

        tool_name = tool_call.get("name")
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return f"The requested tool '{tool_name}' is not available."

        try:
            arguments = self.process_tool_arguments(
                tool_name, 
                tool_call.get("arguments", {})
            )
            result = tool.execute(**arguments)
            formatted_result = self.format_tool_result(result)
            
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response.choices[0].message.content},
                    {"role": "user", "content": f"Tool result: {formatted_result}"}
                ]
            )
            return final_response.choices[0].message.content
            
        except Exception as e:
            return f"An error occurred while executing the tool: {e}"