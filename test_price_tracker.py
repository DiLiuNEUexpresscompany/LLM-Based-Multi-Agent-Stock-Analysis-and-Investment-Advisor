from agents.tool_registry import ToolRegistry
from agents.price_tracker import PriceTracker
from tools.stock_price_tool import StockPriceTool
from tools.data_analysis_tool import DataAnalysisTool

import logging
logging.basicConfig(level=logging.INFO)

registry = ToolRegistry()
registry.register(StockPriceTool())
registry.register(DataAnalysisTool())

agent = PriceTracker(registry)

response = agent.run("What are recent stock prices of Apple")
print(response)
with open("data/price_tracker_output.txt", "w") as f:
    f.write(response)
