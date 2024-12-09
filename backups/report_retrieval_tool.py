# retrieval_tool.py
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from astrapy.client import DataAPIClient
from tools.base_tool import BaseTool
import openai
from openai import OpenAI

class ReportRetrievalTool(BaseTool):
    """Tool to retrieve and process financial reports using vector search"""
    
    def __init__(self, registry=None):
        load_dotenv()
        self.registry = registry
        
        # Fetch required environment variables
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.astra_db_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.astra_db_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.astra_db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
        
        # Validate environment variables
        self._validate_env_vars()
        
        # Set up clients
        openai.api_key = self.api_key
        self.client_openai = OpenAI()
        
        # Connect to the database
        client = DataAPIClient(self.astra_db_token)
        self.database = client.get_database(
            self.astra_db_endpoint, 
            keyspace=self.astra_db_keyspace
        )
        self.collection = self.database.get_collection("financial_report")
    
    def _validate_env_vars(self):
        """Validate that all required environment variables are present"""
        required_vars = {
            "OPENAI_API_KEY": self.api_key,
            "ASTRA_DB_APPLICATION_TOKEN": self.astra_db_token,
            "ASTRA_DB_API_ENDPOINT": self.astra_db_endpoint,
            "ASTRA_DB_KEYSPACE": self.astra_db_keyspace,
        }

        for var_name, value in required_vars.items():
            if not value:
                raise ValueError(f"Missing required environment variable: {var_name}")
    
    def name(self) -> str:
        return "report_retrieval"
    
    def description(self) -> str:
        return """Retrieve financial reports using vector search.
        Arguments:
        - query (required): Search query to find relevant documents
        - limit (optional): Maximum number of documents to retrieve"""
    
    def execute(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve context documents based on vector similarity
        
        :param query: Search query
        :param limit: Maximum number of documents to retrieve
        :return: List of retrieved documents
        """
        try:
            # Perform similarity search
            results = self.collection.find(
                sort={"$vectorize": query},
                limit=limit,
                projection={"$vectorize": True},
                include_similarity=True,
            )
            
            # Process and return results
            vector_texts = []
            for document in results:
                if "$vectorize" in document:
                    vector_texts.append({
                        "text": document["$vectorize"],
                        "similarity": document.get("$similarity", 0)
                    })
            
            return vector_texts
        
        except Exception as e:
            return [{"error": f"Retrieval failed: {str(e)}"}]
