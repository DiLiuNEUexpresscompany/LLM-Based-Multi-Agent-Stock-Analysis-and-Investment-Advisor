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

company = "Apple"
input = f"""
    Analyze recent stock price data for {company}.
    Focus on:
    - Short-term price movements (daily, weekly trends).
    - Key support and resistance levels.
    - Volume trends and market activity.
    - Any potential patterns, such as breakouts or pullbacks.

    Your analysis must:
    - Provide a summary of the stock's current price trends.
    - Include visual charts or tables to highlight price movements.
    - Offer insights into potential trading opportunities based on the data.

    Selected Company: {company}
"""

response = agent.run(input)
print(response)
with open("data/price_tracker_output.txt", "w") as f:
    f.write(response)
