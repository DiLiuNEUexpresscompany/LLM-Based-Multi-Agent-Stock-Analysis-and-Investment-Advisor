import json
import logging
from typing import Dict

# Import necessary classes
from agents.tool_registry import ToolRegistry
from agents.investment_advisor import InvestmentAdvisor
from tools.news_search_tool import NewsSearchTool
from tools.stock_price_tool import StockPriceTool
# from tools.financial_report_tool import FinancialReportTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mock_data(company: str) -> Dict:
    """
    Generate mock data for testing the Investment Advisor
    
    Args:
        company (str): Company ticker symbol
    
    Returns:
        Dict: Comprehensive mock data for investment analysis
    """
    return {
        'news_data': {
            'company': company,
            'sentiment': 'Positive',
            'sentiment_score': 7,
            'headlines': [
                f"{company} Reports Strong Q4 Earnings",
                f"Analysts Bullish on {company}'s Future Prospects",
                f"New Product Launch Signals Growth for {company}"
            ],
            'negative_headlines': [
                "Potential Supply Chain Disruptions",
                "Increased Competition in Market"
            ],
            'upcoming_events': [
                "Annual Shareholder Meeting",
                "Q1 Earnings Report"
            ]
        },
        'price_data': {
            'company': company,
            'current_price': 150.50,
            'trend': 'Upward',
            'trend_score': 6,
            'support_level': 145.00,
            'resistance_level': 155.00,
            'volatility': 0.3,
            'potential_breakout': True
        },
        'report_data': {
            'company_name': f"{company} Corporation",
            'revenue': 50000000,
            'net_income': 10000000,
            'pe_ratio': 22.5,
            'eps': 5.75,
            'financial_health_score': 8,
            'debt_ratio': 0.4,
            'growth_sectors': ['Technology', 'Innovation']
        }
    }

def test_investment_advisor():
    """
    Test the Investment Advisor agent with mock data
    """
    try:
        # Initialize tool registry
        registry = ToolRegistry()
        
        # Optional: Register tools if needed (though the agent doesn't use tools directly)
        # registry.register(NewsSearchTool())
        # registry.register(StockPriceTool())
        # registry.register(FinancialReportTool())
        
        # Create Investment Advisor agent
        advisor = InvestmentAdvisor(registry)
        
        # Test with multiple companies
        test_companies = ['AAPL', 'GOOGL', 'MSFT']
        
        for company in test_companies:
            logger.info(f"Testing Investment Advisor for {company}")
            
            # Generate mock data
            mock_data = generate_mock_data(company)
            
            # Convert data to JSON string
            input_data = json.dumps(mock_data)
            
            # Run investment analysis
            recommendation = advisor.run(input_data)
            
            # Log the recommendation
            logger.info(f"\n{'='*50}\nInvestment Recommendation for {company}\n{'-'*50}")
            print(recommendation)
            logger.info(f"{'='*50}\n")
            
            # Basic validation
            assert recommendation is not None, f"No recommendation generated for {company}"
            assert "Investment Recommendation Report" in recommendation, f"Invalid recommendation format for {company}"
            assert "Disclaimer:" in recommendation, f"Missing disclaimer for {company}"
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

def test_invalid_input():
    """
    Test the Investment Advisor with invalid input
    """
    try:
        # Initialize tool registry
        registry = ToolRegistry()
        
        # Create Investment Advisor agent
        advisor = InvestmentAdvisor(registry)
        
        # Test with invalid JSON
        invalid_inputs = [
            "Not a JSON string",
            json.dumps({"incomplete_data": True}),
            ""
        ]
        
        for invalid_input in invalid_inputs:
            logger.info(f"Testing invalid input: {invalid_input}")
            
            # Run with invalid input
            result = advisor.run(invalid_input)
            
            # Validate error handling
            assert "Invalid input format" in result or "An error occurred" in result, \
                f"Unexpected response for invalid input: {invalid_input}"
    
    except Exception as e:
        logger.error(f"Invalid input test failed: {e}")
        raise

def main():
    """
    Run all tests for the Investment Advisor
    """
    test_investment_advisor()
    # test_invalid_input()
    print("All tests completed successfully!")

if __name__ == "__main__":
    main()