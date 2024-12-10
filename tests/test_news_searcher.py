from tools.news_search_tool import  NewsSearchTool

from agents.news_searcher import NewsSearcher
from agents.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.register(NewsSearchTool())
news_agent = NewsSearcher(registry)


company = "Apple"
input=f"""
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

response = news_agent.run(f"{company} stock")
print("=" * 100)
print(response)

with open("data/news_output.txt", "w") as f:
    f.write(response)