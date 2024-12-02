from tools.news_search import  NewsSearchTool
from tools.stock_price import PolygonStockTool
from agents.news_agent import NewsAgent
from agents.stock_price_agent import StockPriceAgent

from agents.tool_registry import ToolRegistry

def main():
    # registry = ToolRegistry()
    # registry.register(NewsSearchTool())
    # news_agent = NewsAgent(registry)
    

    registry_2 = ToolRegistry()
    registry_2.register(PolygonStockTool())
    stock_agent=StockPriceAgent(registry_2)


    #Example of running a query:
    # response = news_agent.run("Search for some Meta stock news")
    # print(response)
    response_2 = stock_agent.run("Search for 2024-11-20 Apple stock price")
    print(response_2)

if __name__ == "__main__":
    main()