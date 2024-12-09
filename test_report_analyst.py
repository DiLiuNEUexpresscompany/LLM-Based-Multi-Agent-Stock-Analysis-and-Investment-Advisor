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
    question = "What percentage of global electricity usage does NVIDIA aim to match with renewable energy by the end of fiscal year 2025?"
    
    # Run the analysis
    result = report_analyst.run(question)
    
    # Print results
    print("=" * 100)
    print("Question:", result.get('question', 'N/A'))
    print("\nContext:")
    print(result.get('context', 'No context available'))
    print("\n" + "=" * 100)
    print("\nAnalysis:")
    print(result.get('analysis', 'No analysis available'))
    print("\n" + "=" * 100)
    
    # Save results to a file
    with open('data/report_analysis_output.json', 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()