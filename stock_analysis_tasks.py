from tasks.task_base import Task
from textwrap import dedent


class StockAnalysisTasks:
    def research(self, agent, company):
        """Task for researching market news and summaries."""
        return Task(
            description=dedent(f"""
                Collect and summarize recent news articles, publications, 
                and market analyses related to the stock and its industry.
                Focus on significant events, market sentiment, analyst opinions, 
                and upcoming events like earnings releases.

                Your final answer must include:
                - A comprehensive summary of the latest news.
                - Notable shifts in market sentiment.
                - An evaluation of how these developments might impact the stock.
                - Ensure to include the current stock price and a brief trend analysis.

                Use the latest data available to ensure accuracy.

                Selected Company: {company}
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Comprehensive market research summary with stock price insights",
        )

    def financial_analysis(self, agent, company):
        """Task for analyzing the financial health of the company."""
        return Task(
            description=dedent(f"""
                Conduct an in-depth financial analysis of the stock's health and performance.
                Examine key financial metrics, including:
                - P/E ratio
                - EPS growth
                - Revenue trends
                - Debt-to-equity ratio
                Compare the stock's performance with its industry peers and overall market trends.

                Your final report must:
                - Provide a clear assessment of the company's financial health.
                - Highlight strengths, weaknesses, and opportunities in the competitive landscape.
                - Suggest potential risks or advantages based on current financial indicators.

                Ensure the use of the most up-to-date data.

                Selected Company: {company}
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Detailed financial analysis report with key metrics and insights",
        )

    def filings_analysis(self, agent, company):
        """Task for analyzing SEC filings of the company."""
        return Task(
            description=dedent(f"""
                Analyze the latest 10-Q and 10-K filings for the stock using the EDGAR database.
                Focus on:
                - Management's discussion and analysis.
                - Financial statements.
                - Insider trading activity.
                - Risk disclosures and other relevant sections.

                Extract insights that could impact the stock's future performance. 

                Your final report must:
                - Highlight key findings from the filings.
                - Include any red flags or positive indicators that may influence investor decisions.

                Use the latest filings to ensure relevance.

                Selected Company: {company}
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Comprehensive SEC filings analysis report with key takeaways",
        )

    def recommend(self, agent, company):
        """Task for generating investment recommendations."""
        return Task(
            description=dedent(f"""
                Review and synthesize the insights provided by the financial analyst and research analyst.
                Combine these findings to develop a comprehensive investment recommendation.

                Consider:
                - Financial health and market sentiment.
                - Qualitative insights from SEC filings.
                - Upcoming events like earnings releases.
                - Insider trading activity.

                Your final answer must:
                - Be a highly detailed report.
                - Provide a clear investment stance and strategy.
                - Include supporting evidence and visually appealing, well-formatted tables.

                Selected Company: {company}
                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Comprehensive investment recommendation report with strategic insights",
        )

    def __tip_section(self):
        """Returns a motivational tip to encourage high-quality work."""
        return "If you deliver exceptional work, you may receive a $10,000 commission!"
