from tools.news_search_tool import NewsSearchTool
from tools.stock_price_tool import StockPriceTool
from agents.news_finder import NewsFinder
from agents.stock_price_agent import StockPriceAgent
from agents.tool_registry import ToolRegistry


class StockAnalysisSystem:
    def __init__(self):
        # Initialize tool registries and agents
        self.news_registry = ToolRegistry()
        self.news_registry.register(NewsSearchTool())
        self.news_agent = NewsFinder(self.news_registry)

        self.stock_price_registry = ToolRegistry()
        self.stock_price_registry.register(StockPriceTool())
        self.stock_price_agent = StockPriceAgent(self.stock_price_registry)

    def analyze_company(self, company_name):
        # Collect news related to the company
        print(f"Fetching news for {company_name}...\n")
        news_response = self.news_agent.run(f"Search for recent news about {company_name}")
        print("News Results:")
        print(news_response)

        # Fetch the stock price of the company
        print(f"\nFetching stock price for {company_name}...\n")
        stock_response = self.stock_price_agent.run(f"What is the current stock price of {company_name}?")
        print("Stock Price Results:")
        print(stock_response)

        # Combine results into a summary
        summary = f"""
        ### Analysis Report for {company_name} ###

        **News Insights:**
        {news_response}

        **Stock Price:**
        {stock_response}
        """
        print("\n\n########################")
        print("## Analysis Report")
        print("########################\n")
        print(summary)


def main():
    print("Welcome to the Stock Analysis System")
    print('-' * 40)

    company = input("Enter the name of the company you want to analyze: ")

    analysis_system = StockAnalysisSystem()
    analysis_system.analyze_company(company)


if __name__ == "__main__":
    main()
