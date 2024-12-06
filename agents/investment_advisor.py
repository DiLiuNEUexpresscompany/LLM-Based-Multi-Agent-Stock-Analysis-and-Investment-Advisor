# investment_advisor_agent.py
from typing import Dict
from .base_agent import BaseAgent
import json

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

    def get_system_prompt(self, system_prompt = None) -> str:
        if system_prompt is not None:
            return system_prompt
        
        return """
        You are an expert investment advisor tasked with providing comprehensive stock investment analysis. 
        Your role is to synthesize complex financial information and deliver a clear, actionable investment recommendation.

        When analyzing the provided information, you must:
        1. Critically evaluate all provided data, including:
           - Recent news and market sentiment
           - Stock price trends and technical analysis
           - Financial report insights and key performance metrics

        2. Construct a comprehensive investment recommendation that includes:
           - Clear investment stance (Buy/Hold/Sell)
           - Detailed rationale supporting the recommendation
           - Potential risks and opportunities
           - Suggested investment strategy

        3. Provide a well-structured, professional report that:
           - Is easy to understand
           - Includes visual language to enhance comprehension
           - Offers actionable insights for investors
           - Demonstrates a balanced and objective analysis

        4. Consider both short-term and long-term investment perspectives
        5. Use financial terminology appropriately, but ensure clarity for a broad audience
        6. Highlight the most critical insights that could influence investment decisions

        Your analysis should be thorough, nuanced, and provide genuine value to potential investors.
        """

    def process_tool_arguments(self, tool_name: str, arguments: Dict) -> Dict:
        # No tool arguments needed for this agent
        return {}

    def format_investment_recommendation(self, news_data: Dict, price_data: Dict, report_data: Dict) -> str:
        """
        Format the investment recommendation by synthesizing information from different sources.
        
        Args:
            news_data (Dict): Processed news and market sentiment information
            price_data (Dict): Stock price trend and technical analysis
            report_data (Dict): Financial report and performance metrics
        
        Returns:
            str: Comprehensive investment recommendation report
        """
        try:
            # Validate input data
            if not all([news_data, price_data, report_data]):
                return "Insufficient data for comprehensive analysis."

            # Determine investment stance
            investment_stance = self._determine_investment_stance(
                news_data, price_data, report_data
            )

            # Construct detailed report
            report = f"""
                # Investment Recommendation Report

                ## Company Overview
                {report_data.get('company_name', 'Unnamed Company')}

                ## Investment Stance: {investment_stance['recommendation']}

                ### Market Sentiment
                {self._analyze_market_sentiment(news_data)}

                ### Stock Price Analysis
                {self._analyze_price_trends(price_data)}

                ### Financial Performance
                {self._analyze_financial_metrics(report_data)}

                ## Investment Rationale
                {investment_stance['rationale']}

                ## Recommended Strategy
                {investment_stance['strategy']}

                ### Potential Risks
                {self._identify_risks(news_data, price_data, report_data)}

                ### Additional Insights
                {self._generate_additional_insights(news_data, price_data, report_data)}

                *Disclaimer: This recommendation is based on available information and should not be considered absolute financial advice. Always consult with a licensed financial advisor before making investment decisions.*
                """
            return report

        except Exception as e:
            return f"Error generating investment recommendation: {str(e)}"

    def _determine_investment_stance(self, news_data: Dict, price_data: Dict, report_data: Dict) -> Dict:
        """
        Determine investment recommendation based on comprehensive analysis.
        
        Returns:
            Dict with recommendation, rationale, and strategy
        """
        # Implement complex logic to determine recommendation
        # This is a simplified example
        overall_score = 0

        # Market sentiment score
        sentiment_score = news_data.get('sentiment_score', 0)
        overall_score += sentiment_score

        # Price trend score
        price_trend_score = price_data.get('trend_score', 0)
        overall_score += price_trend_score

        # Financial health score
        financial_score = report_data.get('financial_health_score', 0)
        overall_score += financial_score

        # Determine stance based on cumulative score
        if overall_score > 7:
            return {
                'recommendation': 'Strong Buy',
                'rationale': 'Exceptional market conditions, strong financial performance, and positive price trends.',
                'strategy': 'Consider aggressive long-term investment with potential for significant returns.'
            }
        elif overall_score > 4:
            return {
                'recommendation': 'Hold/Moderate Buy',
                'rationale': 'Stable performance with moderate growth potential.',
                'strategy': 'Maintain current position or make gradual incremental investments.'
            }
        else:
            return {
                'recommendation': 'Sell/Avoid',
                'rationale': 'Challenging market conditions and potential downside risks.',
                'strategy': 'Consider reducing exposure or exploring alternative investment opportunities.'
            }

    def _analyze_market_sentiment(self, news_data: Dict) -> str:
        """Analyze and summarize market sentiment from news data."""
        sentiment = news_data.get('sentiment', 'Neutral')
        key_headlines = news_data.get('headlines', [])
        
        return f"""
            - Overall Sentiment: {sentiment}
            - Key Headlines:
            {chr(10).join([f"  * {headline}" for headline in key_headlines[:3]])}
            """

    def _analyze_price_trends(self, price_data: Dict) -> str:
        """Analyze and summarize stock price trends."""
        current_price = price_data.get('current_price', 'N/A')
        trend = price_data.get('trend', 'Stable')
        
        return f"""
            - Current Price: ${current_price}
            - Price Trend: {trend}
            - Key Technical Indicators:
            * Support Level: {price_data.get('support_level', 'N/A')}
            * Resistance Level: {price_data.get('resistance_level', 'N/A')}
            """

    def _analyze_financial_metrics(self, report_data: Dict) -> str:
        """Analyze and summarize key financial metrics."""
        return f"""
            - Revenue: ${report_data.get('revenue', 'N/A')}
            - Net Income: ${report_data.get('net_income', 'N/A')}
            - P/E Ratio: {report_data.get('pe_ratio', 'N/A')}
            - Earnings Per Share (EPS): ${report_data.get('eps', 'N/A')}
            """

    def _identify_risks(self, news_data: Dict, price_data: Dict, report_data: Dict) -> str:
        """Identify potential investment risks."""
        risks = []
        
        # Check news for potential risks
        if news_data.get('negative_headlines'):
            risks.extend(news_data['negative_headlines'][:2])
        
        # Check price volatility
        if price_data.get('volatility', 0) > 0.5:
            risks.append("High price volatility")
        
        # Check financial risks
        if report_data.get('debt_ratio', 0) > 0.6:
            risks.append("High debt levels")
        
        return f"""
            - {chr(10).join([f'* {risk}' for risk in risks]) if risks else 'No significant risks identified'}
            """

    def _generate_additional_insights(self, news_data: Dict, price_data: Dict, report_data: Dict) -> str:
        """Generate additional investment insights."""
        insights = []
        
        # Add potential insights based on data
        if news_data.get('upcoming_events'):
            insights.append(f"Upcoming Events: {', '.join(news_data['upcoming_events'][:2])}")
        
        if price_data.get('potential_breakout'):
            insights.append("Potential price breakout detected")
        
        if report_data.get('growth_sectors'):
            insights.append(f"Growth Sectors: {', '.join(report_data['growth_sectors'])}")
        
        return f"""
            - {chr(10).join([f'* {insight}' for insight in insights]) if insights else 'No additional insights at this time'}
            """

    def run(self, user_input: str) -> str:
        """
        Override the base run method to process complex input.
        
        Expected input is a JSON string containing:
        - news_data: Market news and sentiment
        - price_data: Stock price trends
        - report_data: Financial report analysis
        """
        try:
            # Parse input data
            input_data = json.loads(user_input)
            
            # Extract data components
            news_data = input_data.get('news_data', {})
            price_data = input_data.get('price_data', {})
            report_data = input_data.get('report_data', {})
            
            # Generate investment recommendation
            recommendation = self.format_investment_recommendation(
                news_data, price_data, report_data
            )
            
            return recommendation
        
        except json.JSONDecodeError:
            return "Invalid input format. Please provide a valid JSON string with news, price, and report data."
        except Exception as e:
            return f"An error occurred while generating investment advice: {str(e)}"