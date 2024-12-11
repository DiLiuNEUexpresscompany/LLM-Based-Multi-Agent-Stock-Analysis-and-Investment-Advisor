from agents.tool_registry import ToolRegistry
from agents.price_tracker import PriceTracker
from tools.stock_price_tool import StockPriceTool
from tools.data_analysis_tool import DataAnalysisTool

import logging
import time
import psutil

logging.basicConfig(level=logging.INFO)

# Register tools
registry = ToolRegistry()
registry.register(StockPriceTool())
registry.register(DataAnalysisTool())

# Initialize the agent
agent = PriceTracker(registry)

# Input company and request
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

# Start tracking performance
start_time = time.time()

# Track CPU and memory usage before execution
cpu_before = psutil.cpu_percent(interval=None)
memory_before = psutil.virtual_memory().used

# Try tracking GPU usage if supported
try:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_usage_before = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
except Exception:
    gpu_usage_before = "GPU tracking not available"

# Run the agent
response = agent.run(input)

# End tracking performance
end_time = time.time()
response_time = end_time - start_time

# Track CPU and memory usage after execution
cpu_after = psutil.cpu_percent(interval=None)
memory_after = psutil.virtual_memory().used

# Try tracking GPU usage after execution
try:
    gpu_usage_after = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
except Exception:
    gpu_usage_after = "GPU tracking not available"

# Output response
print(response)
with open("data/test_price_tracker_output.md", "w") as f:
    f.write(response)

# Print performance metrics
print("\nPerformance Metrics:")
print(f"Response Time: {response_time:.2f} seconds")
print(f"CPU Usage Before: {cpu_before}%")
print(f"CPU Usage After: {cpu_after}%")
print(f"Memory Usage Before: {memory_before / (1024 ** 3):.2f} GB")
print(f"Memory Usage After: {memory_after / (1024 ** 3):.2f} GB")
print(f"GPU Usage Before: {gpu_usage_before}")
print(f"GPU Usage After: {gpu_usage_after}")
