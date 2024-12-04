from agents.tool_registry import ToolRegistry
from agents.daily_stock_analysis_agent import DataAnalysisAgent
from tools.stock_price import PolygonStockTool
from tools.enhance_stock_analysis import StockAnalysisTool

import logging
logging.basicConfig(level=logging.INFO)

# 初始化
registry = ToolRegistry()
registry.register(PolygonStockTool())
registry.register(StockAnalysisTool())

# 创建增强版agent
agent = DataAnalysisAgent(registry)

# 分析股票
response = agent.run("Analyze yesterday's trading activity for AAPL")
print(response)
