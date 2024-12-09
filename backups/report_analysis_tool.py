# report_analysis_tool.py
import os
from dotenv import load_dotenv
from astrapy.client import DataAPIClient
from tools.base_tool import BaseTool
from openai import OpenAI

class ReportAnalysisTool(BaseTool):
    """Tool to analyze retrieved financial reports using OpenAI"""
    
    def __init__(self, registry=None):
        load_dotenv()
        self.registry = registry
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def name(self) -> str:
        return "llm_report_analyzer"
    
    def description(self) -> str:
        return """Analyze financial reports using OpenAI.
        Arguments:
        - context_docs (required): Retrieved context documents
        - query (required): Original search query
        - analysis_instructions (optional): Specific analysis guidance"""
    
    def execute(self, context_docs, query, analysis_instructions=None):
        """
        Execute financial report analysis
        
        :param context_docs: List of context documents
        :param query: Original search query
        :param analysis_instructions: Additional analysis guidance
        :return: Analysis result
        """
        try:
            # Combine context texts
            context_text = "\n---\n".join(
                [doc['text'] for doc in context_docs if 'text' in doc]
            )
            
            # Create prompt
            prompt = f"""
            You are an expert-level financial report analysis assistant. Your goal is to carefully examine the provided excerpts and deliver a highly accurate, insightful, and contextually rich summary.

            Query: {query}

            Context:
            {context_text}

            {f"Additional Analysis Instructions: {analysis_instructions}" if analysis_instructions else ""}

            Instructions:
            1. Read the provided excerpts closely to identify all relevant financial metrics, trends, and forward-looking statements.
            2. Summarize these findings in a concise, yet comprehensive manner.
            3. Reflect on your interpretation and potential nuances.
            4. Present the final summary in a clear, professional tone.
            """
            
            # Generate analysis
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful financial analysis assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            
            return {
                "summary": response.choices[0].message.content,
                "context_docs": context_docs
            }
        
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "context_docs": context_docs
            }