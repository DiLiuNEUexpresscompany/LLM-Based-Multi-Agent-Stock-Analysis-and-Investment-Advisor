from agents.tool_registry import ToolRegistry
from agents.price_tracker import PriceTracker
from tools.stock_price_tool import StockPriceTool
from tools.data_analysis_tool import DataAnalysisTool

import logging
logging.basicConfig(level=logging.INFO)

# 初始化
registry = ToolRegistry()
registry.register(StockPriceTool())
registry.register(DataAnalysisTool())

# 创建增强版agent
agent = PriceTracker(registry)

# 分析股票
response = agent.run("Analyze yesterday's trading activity for AAPL")
print(response)
