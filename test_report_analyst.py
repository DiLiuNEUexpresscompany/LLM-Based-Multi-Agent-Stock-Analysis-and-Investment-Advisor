# test_report_analyst.py
import json
import os
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
    
    # Run the analysis
    result = report_analyst.run(input)
    
    # Print results
    # print("=" * 100)
    # print("Question:", result.get('question', 'N/A'))
    # print("\nContext:")
    # print(result.get('context', 'No context available'))
    # print("\n" + "=" * 100)
    # print("\nAnalysis:")
    # print(result.get('analysis', 'No analysis available'))
    # print("\n" + "=" * 100)
    
    print(result)
    # Save results to a file
    with open('data/report_analysis_output.txt', 'w') as f:
        f.write(result)

if __name__ == "__main__":
    main()