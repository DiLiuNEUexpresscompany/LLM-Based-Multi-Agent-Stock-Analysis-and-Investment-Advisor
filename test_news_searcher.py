from tools.news_search_tool import  NewsSearchTool

from agents.news_searcher import NewsSearcher
from agents.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.register(NewsSearchTool())
news_agent = NewsSearcher(registry)

response = news_agent.run("Nvidia stock")
print("=" * 100)
print(response)

with open("data/news_output.txt", "w") as f:
    f.write(response)