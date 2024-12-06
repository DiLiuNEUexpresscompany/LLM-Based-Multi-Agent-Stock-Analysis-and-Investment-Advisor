from tools.news_search_tool import  NewsSearchTool

from agents.news_finder import NewsFinder
from agents.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.register(NewsSearchTool())
news_agent = NewsFinder(registry)

response = news_agent.run("Search for the latest news about Nvidia stock")
print("=" * 100)
print(response)