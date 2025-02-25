# test_news_searcher.py
import time
import psutil
import os

from tools.news_search_tool import NewsSearchTool
from agents.news_searcher import NewsSearcher
from agents.tool_registry import ToolRegistry

# Initialize the tool and registry
registry = ToolRegistry()
registry.register(NewsSearchTool())
news_agent = NewsSearcher(registry)

company = "Alibaba"
input = f"""
    Research and gather the latest news related to {company}.
    Focus on:
    - Significant events affecting the company or industry.
    - Market sentiment and any trending news stories.
    - Analyst opinions and their potential impact on stock performance.
    - Upcoming events like product launches, partnerships, or earnings reports.

    Your final report must:
    - Provide a comprehensive summary of recent and relevant news.
    - Highlight how market sentiment could influence stock performance.
    - Include at least 3 notable headlines with brief summaries.
    - Add any insights into how these developments might impact the stock.

    Selected Company: {company}
"""

# Measure response time
start_time = time.time()

# Get CPU and memory usage before execution
cpu_before = psutil.cpu_percent(interval=None)
memory_before = psutil.virtual_memory().used

# Run the news agent
response = news_agent.run(f"{company} stock")

# Measure response time
end_time = time.time()
response_time = end_time - start_time

# Get CPU and memory usage after execution
cpu_after = psutil.cpu_percent(interval=None)
memory_after = psutil.virtual_memory().used

# Calculate GPU usage if possible
try:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
except Exception as e:
    gpu_usage = "GPU tracking not available"

# Print response
print("=" * 100)
print(response)

# Save response to file
with open("data/test_news_searcher_output.md", "w") as f:
    f.write(response)

# Print performance metrics
print("*" * 100)
print("\nPerformance Metrics:")
print(f"Response Time: {response_time:.2f} seconds")
print(f"CPU Usage Before: {cpu_before}%")
print(f"CPU Usage After: {cpu_after}%")
print(f"Memory Usage Before: {memory_before / (1024 ** 3):.2f} GB")
print(f"Memory Usage After: {memory_after / (1024 ** 3):.2f} GB")
print(f"GPU Usage: {gpu_usage}% (if applicable)")
