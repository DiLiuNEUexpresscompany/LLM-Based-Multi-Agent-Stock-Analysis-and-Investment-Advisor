# tools/report_retrieval_tool.py
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from astrapy.client import DataAPIClient
from tools.base_tool import BaseTool

class ReportRetrievalTool(BaseTool):
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Fetch required environment variables
        self.astra_db_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.astra_db_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.astra_db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

        # Validate environment variables
        required_vars = {
            "ASTRA_DB_APPLICATION_TOKEN": self.astra_db_token,
            "ASTRA_DB_API_ENDPOINT": self.astra_db_endpoint,
            "ASTRA_DB_KEYSPACE": self.astra_db_keyspace,
        }

        for var_name, value in required_vars.items():
            if not value:
                raise ValueError(f"Missing required environment variable: {var_name}")

        # Connect to the Astra database
        client = DataAPIClient(self.astra_db_token)
        self.database = client.get_database(self.astra_db_endpoint, keyspace=self.astra_db_keyspace)
        self.collection = self.database.get_collection("financial_report")

    def name(self) -> str:
        return "report_retrieval_tool"

    def description(self) -> str:
        return "Retrieves financial reports from the Astra vector database based on a query"

    def run(self, query: str) -> Dict[str, Any]:
        """
        Perform a vector similarity search on financial reports
        
        Args:
            query (str): The query to search for in the financial reports
        
        Returns:
            Dict containing the retrieved context and metadata
        """
        try:
            # Perform a similarity search
            results = self.collection.find(
                sort={"$vectorize": query},
                limit=5,
                projection={"$vectorize": True},
                include_similarity=True,
            )

            # Convert results to a list and extract documents
            results_list = list(results)

            # Extract vector texts
            vector_texts = []
            retrieved_docs = []
            for document in results_list:
                if "$vectorize" in document:
                    vector_texts.append(document["$vectorize"])
                    # Create a sanitized version of the document for serialization
                    retrieved_docs.append({
                        k: str(v) for k, v in document.items() 
                        if k != "_id"  # Exclude non-serializable ObjectId
                    })

            # Combine context text
            context_text = "\n---\n".join(vector_texts)

            return {
                "question": query,
                "context": context_text,
                "retrieved_documents": retrieved_docs
            }
        except Exception as e:
            print(f"Error in report retrieval: {e}")
            return {
                "question": query,
                "context": "",
                "retrieved_documents": [],
                "error": str(e)
            }