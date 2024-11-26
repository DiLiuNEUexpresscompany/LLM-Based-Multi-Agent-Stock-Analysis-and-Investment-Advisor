import json
from typing import Dict, Optional
from .base_agent import BaseAgent
class NewsAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        tools_info = self.registry.get_tools_info()
        return f"""
        You are a news search assistant. You can help find and analyze news articles.
        
        Available tools:
        {tools_info}
        
        Call tools using format:
        <tool_call>{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}</tool_call>
        """
    
    def process_tool_arguments(self, tool_name: str, arguments: Dict) -> Dict:
        if tool_name == "search_hackernews":
            # Handle keyword/query parameter mapping
            if "keyword" in arguments:
                arguments["query"] = arguments.pop("keyword")
            # Validate required parameters
            if "query" not in arguments:
                raise ValueError("Search query is required")
        return arguments
    
    def format_tool_result(self, result: Dict) -> str:
        # Custom formatting for news results if needed
        return json.dumps(result)
