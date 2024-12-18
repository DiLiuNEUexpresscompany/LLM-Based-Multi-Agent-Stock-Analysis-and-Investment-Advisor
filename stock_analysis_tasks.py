from crews.task import Task

class StockAnalysisTasks:
    def news_search_task(self, agent, company):
        """Task for finding recent stock news about the company."""
        return Task(
            agent_name="news_searcher",
            description=f"""
                {company} stock
            """,
            agent=agent,
            expected_output="A comprehensive and concise summary of the latest news related to the company, with market sentiment insights.",
        )

    def price_track_task(self, agent, company):
        """Task for analyzing the stock price data of the company."""
        return Task(
            agent_name="price_tracker",
            description=f"""
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
            """,
            agent=agent,
            expected_output="Detailed stock price trend analysis with actionable insights.",
        )
    def report_analysis_task(self, agent, company):
        """Task for reading and analyzing the company's financial reports."""
        return Task(
            agent_name="report_analyst",
            description=f"""
                Review the latest financial reports for {company}, including earnings reports, SEC filings, and annual reports.
                Focus on:
                - Revenue, net income, and cash flow trends.
                - Key financial ratios (e.g., P/E ratio, EPS, ROE).
                - Management's discussion and analysis.
                - Identifying risks, opportunities, or red flags in the filings.

                Your final report must:
                - Provide a detailed breakdown of the company's financial performance.
                - Highlight key strengths, weaknesses, and growth opportunities.
                - Summarize any critical disclosures or statements that may affect investors' confidence.

                Selected Company: {company}
            """,
            agent=agent,
            expected_output="Detailed financial performance and SEC filings analysis with actionable insights.",
        )

    def investment_advice_task(self, agent, company):
        """Task for providing final investment advice for the company."""
        return Task(
            agent_name="investment_advisor",
            description=f"""
                Synthesize all available information on {company} to deliver comprehensive investment advice.
                Consider:
                - Insights from market news and sentiment.
                - Price trend analysis and trading signals.
                - Financial health and key metrics from financial reports.
                - Risks and opportunities identified in SEC filings.

                Your final recommendation must:
                - Provide a clear and actionable investment stance (Buy, Hold, or Sell).
                - Include a rationale supported by detailed analysis.
                - Suggest potential strategies for investors, such as long-term holding or short-term trading opportunities.

                Deliver your advice in a well-formatted report that is easy to understand and visually engaging.

                Selected Company: {company}
            """,
            agent=agent,
            expected_output="Comprehensive investment advice report with actionable recommendations and supporting analysis.",
        )
