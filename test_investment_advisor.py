# test_investment_advisor.py
import sys
import os
import json
import time
import psutil
from agents.investment_advisor import InvestmentAdvisor
from agents.tool_registry import ToolRegistry

# Add the directory containing the agent to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_json_data(file_path):
    """
    Load JSON data from a file, handling potential parsing errors.
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
    news_data_path = 'data/test_news_searcher_output.md'
    price_data_path = 'data/test_price_tracker_output.md'
    report_data_path = 'data/test_report_analyst_output.md'

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

    # Start tracking performance
    start_time = time.time()

    # Capture system resource usage before execution
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

    # Generate the analysis prompt
    analysis_prompt = advisor.generate_investment_analysis_prompt(
        news_data, 
        price_data, 
        report_data
    )

    # Run the analysis
    investment_report = advisor.run(analysis_prompt)

    # End tracking performance
    end_time = time.time()
    response_time = end_time - start_time

    # Capture system resource usage after execution
    cpu_after = psutil.cpu_percent(interval=None)
    memory_after = psutil.virtual_memory().used

    # Try tracking GPU usage after execution
    try:
        gpu_usage_after = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    except Exception:
        gpu_usage_after = "GPU tracking not available"

    # Print the investment report
    print("\n--- INVESTMENT ANALYSIS REPORT ---")
    print(investment_report)

    # Optionally, save the report to a file
    with open('data/test_investment_advisor_output.md', 'w') as report_file:
        report_file.write(investment_report)

    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"Response Time: {response_time:.2f} seconds")
    print(f"CPU Usage Before: {cpu_before}%")
    print(f"CPU Usage After: {cpu_after}%")
    print(f"Memory Usage Before: {memory_before / (1024 ** 3):.2f} GB")
    print(f"Memory Usage After: {memory_after / (1024 ** 3):.2f} GB")
    print(f"GPU Usage Before: {gpu_usage_before}")
    print(f"GPU Usage After: {gpu_usage_after}")

if __name__ == "__main__":
    main()
