# agents/report_analyst.py
from typing import Dict, Any
from agents.tool_registry import ToolRegistry

class ReportAnalyst:
    def __init__(self, tool_registry: ToolRegistry):
        """
        Initialize the report analyst with a tool registry
        
        Args:
            tool_registry (ToolRegistry): Registry of available tools
        """
        self.tool_registry = tool_registry

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the report analysis workflow
        
        Args:
            query (str): The query to analyze
        
        Returns:
            Dict containing analysis results
        """
        # Get retrieval and analysis tools
        retrieval_tool = self.tool_registry.get_tool("report_retrieval_tool")
        analysis_tool = self.tool_registry.get_tool("report_analysis_tool")

        # Retrieve context
        retrieval_result = retrieval_tool.run(query)

        # Analyze the retrieved context
        analysis_result = analysis_tool.run(
            query=query, 
            context=retrieval_result.get('context', '')
        )

        # Combine results
        combined_result = {
            **retrieval_result,
            **analysis_result
        }

        return combined_result