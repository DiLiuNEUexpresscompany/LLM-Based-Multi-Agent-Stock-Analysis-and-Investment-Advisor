# tools/report_analysis_tool.py
import os
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from textwrap import dedent
from tools.base_tool import BaseTool

class ReportAnalysisTool(BaseTool):
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Fetch required environment variables
        hf_api_key = os.getenv("HUGGUNGFACE_ENDPOINT_API_KEY")

        # Validate environment variables
        if not hf_api_key:
            raise ValueError("Missing required environment variable: HUGGUNGFACE_ENDPOINT_API_KEY")

        # Connect to the Hugging Face TGI client
        self.client_hf = OpenAI(
            base_url="https://b4hv8vle4pryfmy8.us-east-1.aws.endpoints.huggingface.cloud/v1/",
            api_key=hf_api_key
        )

    def name(self) -> str:
        return "report_analysis_tool"

    def description(self) -> str:
        return "Analyzes financial reports using a generative AI model with comprehensive financial analysis instructions"

    def create_analysis_prompt(self, data_row: Dict[str, str]) -> Dict[str, Any]:
        """
        Create a detailed analysis prompt for the financial report
        
        Args:
            data_row (Dict): A dictionary containing question and context
        
        Returns:
            Dict of messages for the AI model
        """
        prompt = dedent(f"""
            You are an expert-level financial report analysis assistant. Your goal is to carefully examine the provided excerpts and deliver a highly accurate, insightful, and contextually rich analysis of the key financial elements mentioned. 
                                
            Question:
            {data_row["question"]}

            Context:
            {data_row["context"]}

            ### Instructions
            1. Key Metrics Analysis
            - Identify critical financial metrics
            - Calculate growth rates and trends

            2. Performance Summary
            - Financial results overview
            - Notable operational trends
            - YoY/QoQ comparisons

            3. Strategic Analysis
            - Operational efficiency
            - Market conditions impact

            4. Risk Assessment
            - Data uncertainties
            - Market risks
            - Strategic challenges

            5. Market Context
            - Macroeconomic factors
            - Inflation impact
            - Interest rate environment
            - Competitive positioning
            - Industry trends

            ### Output Format
            Please provide analysis in flowing paragraphs rather than bullet points covering:
            - Key metrics summary
            - Performance analysis
            - Strategic implications
            - Risk factors
            - Market context
            Ensuring clear transitions between topics while maintaining professional financial reporting standards.
        """)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        return messages

    def run(self, query: str, context: str = None) -> Dict[str, Any]:
        """
        Run the financial report analysis
        
        Args:
            query (str): The query to analyze
            context (str, optional): Context for the analysis. If not provided, retrieval tool should be used.
        
        Returns:
            Dict containing the analysis results
        """
        # Prepare data row for prompt creation
        data_row = {
            "question": query,
            "context": context or "No context provided.",
        }

        # Create prompt using the template
        messages = self.create_analysis_prompt(data_row)

        # Use the Hugging Face TGI model to generate a response
        chat_completion = self.client_hf.chat.completions.create(
            model="tgi",
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )

        # Extract the generated analysis
        analysis = chat_completion.choices[0].message.content

        return {
            "question": query,
            "context": data_row["context"],
            "analysis": analysis
        }