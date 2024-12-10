# investment_advisor_agent.py
from typing import Dict
from .base_agent import BaseAgent
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

class InvestmentAdvisor(BaseAgent):
    def __init__(self, registry):
        super().__init__(registry)
        self.role = "Investment Advisor"
        self.goal = (
            "To synthesize complex financial information from multiple sources and provide "
            "comprehensive, actionable investment advice for a specific company. The advisor aims "
            "to deliver nuanced insights that combine market news, stock price trends, and financial "
            "performance analysis into clear investment recommendations."
        )
        self.backstory = (
            "As a seasoned financial analyst, this agent combines deep analytical skills with "
            "a holistic approach to investment research. Drawing from diverse data sources, "
            "the advisor provides strategic insights that go beyond surface-level information, "
            "helping investors make informed decisions based on comprehensive market intelligence."
        )
        # No tools for this agent
        self.tools = []
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client_OpenAI = OpenAI()

    def process_tool_arguments(self, tool_name, arguments):
        pass
    def get_system_prompt(self, system_prompt=None):
        pass

    def generate_investment_analysis_prompt(self, news_data, price_data, report_data):
        """
        Generate a comprehensive prompt for investment analysis
        
        Args:
        - news_data (dict): Recent news about the company
        - price_data (dict): Recent stock price and technical analysis
        - report_data (dict): Recent financial report insights
        
        Returns:
        - str: Detailed prompt for investment analysis
        """
        prompt = f"""
            You are a senior investment advisor conducting a comprehensive stock analysis. 
            Provide a detailed investment recommendation based on the following data:

            📰 RECENT NEWS ANALYSIS:
            {json.dumps(news_data, indent=2)}

            💹 STOCK PRICE DYNAMICS:
            {json.dumps(price_data, indent=2)}

            📊 FINANCIAL REPORT INSIGHTS:
            {json.dumps(report_data, indent=2)}

            REQUIRED ANALYSIS COMPONENTS:
            1. Company Fundamental Health
            - Assess overall financial stability
            - Evaluate key financial ratios
            - Identify potential growth indicators or red flags

            2. Market Sentiment and External Factors
            - Analyze recent news impact on stock perception
            - Assess industry trends and competitive landscape
            - Evaluate macroeconomic influences

            3. Technical Price Analysis
            - Review recent price movements
            - Identify support and resistance levels
            - Assess momentum and trading signals

            4. Risk Assessment
            - Short-term and long-term risk evaluation
            - Potential volatility factors
            - Comparative risk against sector benchmarks

            5. Investment Recommendation
            - Provide a clear recommendation:
                * Strong Buy
                * Buy
                * Hold
                * Sell
                * Strong Sell
            - Justify recommendation with concrete evidence
            - Suggest potential investment strategy (e.g., long-term hold, swing trade)

            6. Confidence Level
            - Rate your recommendation's confidence (0-100%)
            - Explain key factors influencing confidence

            IMPORTANT GUIDELINES:
            - Be objective and data-driven
            - Avoid sensationalism
            - Clearly distinguish between facts and interpretations
            - Consider both quantitative and qualitative factors

            Deliver the analysis in a professional, concise, and actionable format.
            """
        return prompt
    
    def run(self, prompt):
        response = self.client_OpenAI.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a senior investment advisor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        
        # Extract the generated report
        report_content = response.choices[0].message.content
        return report_content