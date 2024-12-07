# report_analyzer.py
from typing import Any, Dict
from agents.tool_registry import ToolRegistry

class ReportAnalyzer:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    def run(self, query: str, 
            context_length: int = 5, 
            analysis_instructions: str = None) -> Any:
        """
        Run report analysis process
        
        :param query: Input query or topic to analyze
        :param context_length: Number of context documents to retrieve
        :param analysis_instructions: Specific analysis details
        :return: Analysis result
        """
        try:
            # Get the retrieval tool
            retrieval_tool = self.registry.get_tool('report_retrieval')
            
            # Get the context documents
            context_docs = retrieval_tool.execute(
                query=query, 
                limit=context_length
            )
            
            # Get the LLM analysis tool
            analysis_tool = self.registry.get_tool('llm_report_analyzer')
            
            # Execute the analysis
            result = analysis_tool.execute(
                context_docs=context_docs,
                query=query,
                analysis_instructions=analysis_instructions
            )
            
            return result
        
        except Exception as e:
            return {
                "error": f"Report analysis failed: {str(e)}",
                "details": str(e)
            }