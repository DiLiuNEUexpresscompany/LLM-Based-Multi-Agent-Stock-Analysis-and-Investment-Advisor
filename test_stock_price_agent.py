from tools.stock_price_tool import StockPriceTool
from data.stock_price_agent import StockPriceAgent

from agents.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.register(StockPriceTool())
stock_agent=StockPriceAgent(registry)
response = stock_agent.run("What is the price of Tesla in 2024-11-15?")
print(response)