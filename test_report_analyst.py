import json
import os
import time
import psutil
from tools.report_retrieval_tool import ReportRetrievalTool
from tools.report_analysis_tool import ReportAnalysisTool
from agents.report_analyst import ReportAnalyst
from agents.tool_registry import ToolRegistry

def main():
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)

    # Create a tool registry
    registry = ToolRegistry()
    
    # Register the report retrieval and analysis tools
    registry.register(ReportRetrievalTool())
    registry.register(ReportAnalysisTool())
    
    # Create the report analyst agent
    report_analyst = ReportAnalyst(registry)
    
    # Example query
    company = "Apple"
    input = f"""
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
    """

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

    # Run the analysis
    result = report_analyst.run(input)

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

    # Print the result
    print(result)

    # Save results to a file
    with open('data/test_report_analyst_output.md', 'w') as f:
        f.write(result)

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
