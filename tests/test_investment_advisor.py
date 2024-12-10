import sys
import os
import json
from agents.investment_advisor import InvestmentAdvisor
from agents.tool_registry import ToolRegistry

# Add the directory containing the agent to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_json_data(file_path):
    """
    Load JSON data from a file, handling potential parsing errors
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        # If it's not JSON, try reading as text
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

def main():
    # Paths to the data files
    news_data_path = 'data/news_searcher_output.md'
    price_data_path = 'data/price_tracker_output.md'
    report_data_path = 'data/report_analyst_output.md'

    # Load data
    news_data = load_json_data(news_data_path)
    price_data = load_json_data(price_data_path)
    report_data = load_json_data(report_data_path)

    # Check if all data is loaded
    if not all([news_data, price_data, report_data]):
        print("Failed to load one or more data files.")
        return

    # Initialize the Investment Advisor
    registry = ToolRegistry()
    advisor = InvestmentAdvisor(registry)

    # Generate the analysis prompt
    analysis_prompt = advisor.generate_investment_analysis_prompt(
        news_data, 
        price_data, 
        report_data
    )

    # Run the analysis
    investment_report = advisor.run(analysis_prompt)

    # Print the investment report
    print("\n--- INVESTMENT ANALYSIS REPORT ---")
    print(investment_report)

    # Optionally, save the report to a file
    with open('data/investment_analysis_report.md', 'w') as report_file:
        report_file.write(investment_report)

if __name__ == "__main__":
    main()