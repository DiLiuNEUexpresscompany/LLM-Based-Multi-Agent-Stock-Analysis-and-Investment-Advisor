from tools.news_search import NewsSearchTool
from tools.stock_price import PolygonStockTool
from tools.market_analysis_tools import MarketTrendTool, IndustryComparisonTool
from tools.financial_analysis_tools import FinancialStatementAnalysisTool, RiskAssessmentTool
from tools.sec_filing_tools import Retrieve10KTool, Retrieve10QTool
from agents.general_agent import GeneralAgent


class StockAnalysisAgents:
    def market_research_analyst(self):
        """Creates a market research analyst agent."""
        return GeneralAgent(
            role="Market Research Analyst",
            goal="Conduct in-depth market research and uncover hidden investment opportunities.",
            backstory="An experienced analyst with deep insights into market trends and competitive landscapes.",
            verbose=True,
            tools=[
                NewsSearchTool(),
                MarketTrendTool(),
                IndustryComparisonTool(),
            ],
        )

    def financial_strategist(self):
        """Creates a financial strategist agent."""
        return GeneralAgent(
            role="Financial Strategist",
            goal="Provide detailed financial analysis and investment insights.",
            backstory="An expert in financial modeling and strategy, skilled at interpreting complex financial data.",
            verbose=True,
            tools=[
                FinancialStatementAnalysisTool(),
                Retrieve10KTool(),
                Retrieve10QTool(),
            ],
        )

    def investment_advisor(self):
        """Creates an investment advisor agent."""
        return GeneralAgent(
            role="Investment Advisor",
            goal="Offer actionable and comprehensive investment advice.",
            backstory="A seasoned advisor skilled at transforming complex analyses into clear investment strategies.",
            verbose=True,
            tools=[
                RiskAssessmentTool(),
                MarketTrendTool(),
                NewsSearchTool(),
            ],
        )
