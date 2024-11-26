from tools.news_search import  NewsSearchTool
from agents.news_agent import NewsAgent
from agents.tool_registry import ToolRegistry

def main():
    registry = ToolRegistry()
    registry.register(NewsSearchTool())
    # 创建新闻agent
    news_agent = NewsAgent(registry)
    # Example of running a query:
    response = news_agent.run("Search for some Intel stock news")
    print(response)

if __name__ == "__main__":
    main()