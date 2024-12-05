from agents.news_finder import NewsFinder
from agents.price_tracker import PriceTracker
from agents.tool_registry import ToolRegistry
from agents.base_agent import BaseAgent

from tools.stock_price_tool import StockPriceTool
from tools.news_search_tool import  NewsSearchTool
from tools.data_analysis_tool import DataAnalysisTool


class StockAnalysisAgents:
    def news_finder(self):
        """Creates a news finder agent."""
        registry = ToolRegistry()
        registry.register(NewsSearchTool())
        return NewsFinder(registry)

    def price_tracker(self):
        """Creates a price tracker agent."""
        registry = ToolRegistry()
        registry.register(StockPriceTool())
        registry.register(DataAnalysisTool())
        return PriceTracker(registry)
    
    def report_analyzer(self):
        """Creates a report analyzer agent."""
        pass

    def investment_advisor(self):
        """Creates an investment advisor agent."""
        pass
