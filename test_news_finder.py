from tools.news_search_tool import  NewsSearchTool

from agents.news_finder import NewsFinder
from agents.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.register(NewsSearchTool())
news_agent = NewsFinder(registry)

print(dir(news_agent))

response = news_agent.run("Search Meta stock")
print("=" * 200)
print(response)