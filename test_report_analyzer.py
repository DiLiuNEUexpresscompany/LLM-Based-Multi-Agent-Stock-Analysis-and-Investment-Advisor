# test_report_analyzer.py
import os
from tools.retrieval_tool import ReportRetrievalTool
from tools.report_analysis_tool import ReportAnalysisTool
from agents.report_analyzer import ReportAnalyzer
from agents.tool_registry import ToolRegistry

def main():
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)

    # Create a tool registry
    registry = ToolRegistry()
    
    # Register the retrieval and analysis tools
    registry.register(ReportRetrievalTool())
    registry.register(ReportAnalysisTool())
    
    # Create the report analyzer agent
    report_analyzer = ReportAnalyzer(registry)
    
    # Example query
    query = "Analyze nvidia's financial performance over the past year."
    
    # Run analysis
    result = report_analyzer.run(
        query=query,
        context_length=5,
        analysis_instructions="Focus on potential market trends and economic indicators"
    )
    
    # Prepare output content
    output_content = "=" * 100 + "\n"
    output_content += f"Query: {query}\n\n"
    output_content += "Summary:\n" + result.get('summary', 'No summary generated') + "\n\n"
    
    # Add context documents
    output_content += "Context Documents:\n"
    for doc in result.get('context_docs', []):
        output_content += f"  - Similarity: {doc.get('similarity', 'N/A')}\n"
        output_content += f"    Text: {doc.get('text', 'No text')[:200]}...\n"
    output_content += "=" * 100

    # Print to console
    print(output_content)
    
    # Save to file
    with open('data/report_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(output_content)

if __name__ == "__main__":
    main()