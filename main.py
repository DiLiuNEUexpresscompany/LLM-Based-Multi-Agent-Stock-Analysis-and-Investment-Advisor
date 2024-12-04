from tools.news_search_tool import  NewsSearchTool
from tools.stock_price_tool import StockPriceTool
from tools.data_analysis_tool import DataAnalysisTool
from agents.news_finder import NewsFinder
from agents.stock_price_agent import StockPriceAgent
from agents.enhance_stock_analysis_agent import StockAnalysisAgent

from agents.tool_registry import ToolRegistry

def main():
    # registry = ToolRegistry()
    # registry.register(NewsSearchTool())
    # news_agent = NewsAgent(registry)


    # registry_2 = ToolRegistry()
    # registry_2.register(PolygonStockTool())
    # stock_agent=StockPriceAgent(registry_2)

    # registry_news_agent = ToolRegistry()
    # registry_news_agent.register(NewsSearchTool())
    # news_agent = NewsAgent(registry_news_agent)
    # response = news_agent.run("Search for some news about Nvidia stock")
    # print(response)
    
    # response = news_agent.run("Search for some Meta stock news")
    # print(response)
    # response_2 = stock_agent.run("What is the price of Tesla in 2024-11-25?")
    # print(response_2)

    registry = ToolRegistry()
    registry.register(StockPriceTool())
    registry.register(DataAnalysisTool())

    # Create the agent with HuggingFace endpoint
    agent = StockAnalysisAgent(
        registry=registry,
        base_url="https://mqiez28uc19t0z3q.us-east-1.aws.endpoints.huggingface.cloud/v1/",
        api_key="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"
    )

    # Example usage with streaming response
    response = agent.run(
        "Analyze AAPL stock with market comparison and risk metrics", 
        stream=True,
    )
        

if __name__ == "__main__":
    main()