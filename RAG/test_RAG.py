import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from astrapy.db import AstraDB
from astrapy import DataAPIClient

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

class FinancialQASystem:
    def __init__(self, debug: bool = True):
        """
        Initialize the Financial QA System with necessary configurations and connections.
        
        Args:
            debug (bool): Enable/disable debug mode for verbose output
        """
        self.debug = debug
        self._load_environment()
        self._initialize_clients()
        self._setup_chain()

    def _load_environment(self) -> None:
        """Load and validate all required environment variables."""
        load_dotenv()
        
        required_vars = {
            "OPENAI_API_KEY": "OpenAI API key",
            "ASTRA_DB_APPLICATION_TOKEN": "Astra DB application token",
            "ASTRA_DB_API_ENDPOINT": "Astra DB API endpoint",
            "ASTRA_DB_KEYSPACE": "Astra DB keyspace"
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing_vars.append(description)
        
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        self.COLLECTION_NAME = "financial_report_2"

    def _initialize_clients(self) -> None:
        """Initialize OpenAI and AstraDB clients."""
        try:
            self.embeddings = OpenAIEmbeddings(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
            self.db = AstraDB(
                token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
                api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
                namespace=os.getenv("ASTRA_DB_KEYSPACE")
            )
            
            self.collection = self.db.collection(
                collection_name=self.COLLECTION_NAME
            )
            
            if self.debug:
                print("✓ Clients initialized successfully")
                
        except Exception as e:
            raise ConnectionError(f"Failed to initialize clients: {str(e)}")

    def _setup_chain(self) -> None:
        """Set up the LangChain processing chain."""
        prompt_template = """
        Based on the following financial report excerpts, please answer the question.
        If the information is not complete enough in the provided context, please specify what information is missing.

        Context: {context}
        Question: {question}

        Please provide a detailed answer based only on the information present in the context:
        """
        
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        
        self.model = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4",
            temperature=0.1
        )
        
        self.chain = (
            {
                "context": lambda x: x["context"],
                "question": lambda x: x["question"]
            }
            | self.prompt
            | self.model
        )

    def _get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document collection.
        
        Returns:
            Dict containing collection statistics
        """
        try:
            total_docs = self.collection.find({})
            return {
                "total_documents": len(total_docs),
                "collection_name": self.COLLECTION_NAME
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def _retrieve_documents(
        self,
        question: str,
        limit: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a given question.
        
        Args:
            question (str): The question to find relevant documents for
            limit (int): Maximum number of documents to retrieve
            similarity_threshold (float): Minimum similarity score for documents
            
        Returns:
            List of relevant documents
        """
        question_embedding = self.embeddings.embed_query(question)
        
        results = self.collection.vector_find(
            vector=question_embedding,
            limit=limit,
            fields=["$vectorize"],
            include_similarity=True
        )
        
        if self.debug:
            print(f"\nFound {len(results)} documents")
            for idx, doc in enumerate(results):
                print(f"\nDocument {idx + 1} (similarity: {doc.get('$similarity', 'N/A'):.4f}):")
                print(doc.get('$vectorize', '')[:200] + "...")
        
        return [
            doc for doc in results
            if doc.get('$similarity', 0) > similarity_threshold
        ]

    def answer_question(
        self,
        question: str,
        limit: int = 5,
        similarity_threshold: float = 0.5
    ) -> str:
        """
        Process a question and return an answer based on the document collection.
        
        Args:
            question (str): The question to answer
            limit (int): Maximum number of documents to retrieve
            similarity_threshold (float): Minimum similarity score for documents
            
        Returns:
            str: The answer to the question
        """
        try:
            if self.debug:
                stats = self._get_collection_stats()
                print("\nCollection Statistics:")
                print(f"Total Documents: {stats.get('total_documents', 'Unknown')}")
                print(f"Collection Name: {stats.get('collection_name', 'Unknown')}")
                print("\nRetrieving relevant documents...")
            
            relevant_docs = self._retrieve_documents(
                question,
                limit,
                similarity_threshold
            )
            
            if not relevant_docs:
                return "No relevant documents found to answer the question."
            
            # Extract and clean context
            context_docs = [
                doc['$vectorize'].replace("$", " $")
                for doc in relevant_docs
                if '$vectorize' in doc
            ]
            
            # Join documents with clear separation
            context = "\n---\n".join(context_docs)
            
            if self.debug:
                print("\nProcessing query with combined context...")
            
            # Get response
            response = self.chain.invoke({
                "context": context,
                "question": question
            })
            
            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            return str(response)
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            return error_msg

def main():
    """Main function to run the QA system."""
    try:
        # Initialize the QA system
        qa_system = FinancialQASystem(debug=True)
        
        # Example question
        default_question = "Can you summarize the Alphabet Announces Third Quarter 2024 Results?"
        print(f"\nDefault question: {default_question}")
        
        # Get user input
        user_question = input("\nEnter your question (or press Enter to use default): ").strip()
        question = user_question if user_question else default_question
        
        # Get and print answer
        answer = qa_system.answer_question(question)
        print("\nAnswer:")
        print(answer)
        
    except Exception as e:
        print(f"Error running QA system: {e}")

if __name__ == "__main__":
    main()