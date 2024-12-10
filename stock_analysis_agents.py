from agents.news_searcher import NewsSearcher
from agents.price_tracker import PriceTracker
from agents.tool_registry import ToolRegistry
from agents.report_analyst import ReportAnalyst
from agents.investment_advisor import InvestmentAdvisor

from tools.stock_price_tool import StockPriceTool
from tools.news_search_tool import  NewsSearchTool
from tools.data_analysis_tool import DataAnalysisTool
from tools.report_retrieval_tool import ReportRetrievalTool
from tools.report_analysis_tool import ReportAnalysisTool
from backups.format_convert_tool import FormatConvertTool

class StockAnalysisAgents:
    def news_searcher(self):
        """Creates a news finder agent."""
        registry = ToolRegistry()
        registry.register(NewsSearchTool())
        return NewsSearcher(registry)

    def price_tracker(self):
        """Creates a price tracker agent."""
        registry = ToolRegistry()
        registry.register(StockPriceTool())
        registry.register(DataAnalysisTool())
        return PriceTracker(registry)
    
    def report_analyst(self):
        """Creates a report analyzer agent."""
        registry = ToolRegistry()
        registry.register(ReportRetrievalTool())
        registry.register(ReportAnalysisTool())
        return ReportAnalyst(registry)

    def investment_advisor(self):
        """Creates an investment advisor agent."""
        registry = ToolRegistry()
        registry.register(FormatConvertTool())
        return InvestmentAdvisor(registry)
