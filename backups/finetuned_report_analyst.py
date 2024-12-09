import os
from openai import OpenAI
from dotenv import load_dotenv
from astrapy.client import DataAPIClient
from textwrap import dedent

class FinetunedReportAnalyst:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Fetch required environment variables
        self.hf_api_key = os.getenv("HUGGUNGFACE_ENDPOINT_API_KEY")
        self.astra_db_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.astra_db_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.astra_db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

        # Validate environment variables
        self._validate_env_vars()

        # Initialize clients
        self.client_hf = self._init_huggingface_client()
        self.database_client = self._init_database_client()

    def _validate_env_vars(self):
        """Validate that all required environment variables are present."""
        required_vars = {
            "HUGGUNGFACE_ENDPOINT_API_KEY": self.hf_api_key,
            "ASTRA_DB_APPLICATION_TOKEN": self.astra_db_token,
            "ASTRA_DB_API_ENDPOINT": self.astra_db_endpoint,
            "ASTRA_DB_KEYSPACE": self.astra_db_keyspace,
        }

        for var_name, value in required_vars.items():
            if not value:
                raise ValueError(f"Missing required environment variable: {var_name}")

    def _init_huggingface_client(self):
        """Initialize Hugging Face TGI client."""
        return OpenAI(
            base_url="https://b4hv8vle4pryfmy8.us-east-1.aws.endpoints.huggingface.cloud/v1/",
            api_key=self.hf_api_key
        )

    def _init_database_client(self):
        """Initialize Astra database client."""
        client = DataAPIClient(self.astra_db_token)
        database = client.get_database(self.astra_db_endpoint, keyspace=self.astra_db_keyspace)
        return database.get_collection("financial_report")

    def _create_test_prompt(self, question, context):
        """Create a structured prompt for financial report analysis."""
        prompt = dedent(f"""
            You are an expert-level financial report analysis assistant. Your goal is to carefully examine the provided excerpts and deliver a highly accurate, insightful, and contextually rich analysis of the key financial elements mentioned, such as earnings, revenue, guidance, margins, and other performance indicators. 
                                
                Question:
                {question}

                Context:
                {context}

            ### Instructions
            1. Key Metrics Analysis
            - Identify critical financial metrics
            - Calculate growth rates and trends
            - Compare with industry benchmarks

            2. Performance Summary
            - Financial results overview
            - Notable operational trends
            - YoY/QoQ comparisons

            3. Strategic Analysis
            - Operational efficiency
            - Market conditions impact
            - Capital allocation review

            4. Risk Assessment
            - Data uncertainties
            - Market risks
            - Strategic challenges

            5. Market Context
            - Macroeconomic factors
            - Inflation impact
            - Interest rate environment
            - FX exposure
            - Competitive positioning
            - Industry trends

            ### Output Format
            Please provide analysis in flowing paragraphs rather than bullet points covering:
            - Key metrics summary
            - Performance analysis
            - Strategic implications
            - Risk factors
            - Market context
            ensuring clear transitions between topics while maintaining professional financial reporting standards.
        """)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        return messages

    def run(self, question):
        """
        Run the financial report analysis for a given question.
        
        Args:
            question (str): The specific question to analyze
        
        Returns:
            str: Detailed financial analysis response
        """
        # Perform a similarity search
        results = self.database_client.find(
            sort={"$vectorize": question},
            limit=5,
            projection={"$vectorize": True},
            include_similarity=True,
        )

        # Extract context from vector search results
        vector_texts = [
            document.get("$vectorize", "") for document in results
            if "$vectorize" in document
        ]

        # Combine all $vectorize text as context
        context_text = "\n---\n".join(vector_texts)
        if not context_text.strip():
            return "No relevant context found for the given question."

        # Create prompt using the template
        messages = self._create_test_prompt(question, context_text)

        # Generate response using Hugging Face TGI model
        try:
            chat_completion = self.client_hf.chat.completions.create(
                model="tgi",
                messages=messages,
                temperature=0.1,
                max_tokens=512,
                stream=False,
            )

            # Return the generated response
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"An error occurred during analysis: {str(e)}"