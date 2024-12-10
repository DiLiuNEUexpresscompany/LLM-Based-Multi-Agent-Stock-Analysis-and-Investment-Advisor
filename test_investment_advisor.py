import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Configure OpenAI Client
# IMPORTANT: Replace with your actual OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client_OpenAI = OpenAI()

def generate_investment_analysis_prompt(news_data, price_data, report_data):
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

def generate_investment_report(user_input, output_file='data/final_report.txt'):
    """
    Generate an investment report based on input data from multiple agents
    
    Args:
    - user_input (str): JSON-formatted string containing agent data
    - output_file (str): Path to save the generated report
    
    Returns:
    - dict: Investment analysis report details
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Parse input data
        input_data = json.loads(user_input)
        
        # Extract data components
        news_data = input_data.get('news_data', {})
        price_data = input_data.get('price_data', {})
        report_data = input_data.get('report_data', {})
        
        # Generate prompt for LLM
        prompt = generate_investment_analysis_prompt(news_data, price_data, report_data)
        
        # Call OpenAI API
        response = client_OpenAI.chat.completions.create(
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
        
        # Save the report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Investment Analysis Prompt:\n")
            f.write(prompt)
            f.write("\n\n--- Investment Report ---\n")
            f.write(report_content)
        
        # Return report details
        return {
            "prompt": prompt,
            "report": report_content,
            "output_file": output_file,
            "usage": response.usage
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "details": "Failed to generate investment report"
        }

# Example usage
if __name__ == "__main__":
    # Sample input data (replace with your actual data)
    sample_input = json.dumps({
        "news_data": {
            "recent_headlines": [
                "Company X Reports Strong Q2 Earnings",
                "Tech Sector Facing Challenges"
            ],
            "sentiment": "Mostly Positive"
        },
        "price_data": {
            "current_price": 150.25,
            "52_week_range": {"low": 120.50, "high": 180.75},
            "moving_averages": {
                "50_day": 145.60,
                "200_day": 155.30
            }
        },
        "report_data": {
            "revenue": "1.2B",
            "net_income": "250M",
            "key_ratios": {
                "P/E": 22.5,
                "ROE": "15.3%"
            }
        }
    })
    
    # Generate investment report
    report = generate_investment_report(sample_input)
    
    # Print the report
    print("Investment Analysis Prompt:")
    print(report.get("prompt", "No prompt generated"))
    print("\n--- Investment Report ---")
    print(report.get("report", "No report generated"))
    
    # Print token usage if available
    if "usage" in report:
        print("\nToken Usage:")
        print(f"Prompt Tokens: {report['usage'].prompt_tokens}")
        print(f"Completion Tokens: {report['usage'].completion_tokens}")
        print(f"Total Tokens: {report['usage'].total_tokens}")