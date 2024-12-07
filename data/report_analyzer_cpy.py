import os
from dotenv import load_dotenv
from astrapy.client import DataAPIClient
import openai
from openai import OpenAI

def main():
    # Load the environment variables
    load_dotenv()

    # Fetch required environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    astra_db_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    astra_db_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    astra_db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

    # Ensure all required variables are present
    required_vars = {
        "OPENAI_API_KEY": api_key,
        "ASTRA_DB_APPLICATION_TOKEN": astra_db_token,
        "ASTRA_DB_API_ENDPOINT": astra_db_endpoint,
        "ASTRA_DB_KEYSPACE": astra_db_keyspace,
    }

    for var_name, value in required_vars.items():
        if not value:
            raise ValueError(f"Missing required environment variable: {var_name}")

    # Set OpenAI API key
    openai.api_key = api_key
    client_OpenAI = OpenAI()

    # Connect to the database
    client = DataAPIClient(astra_db_token)
    database = client.get_database(astra_db_endpoint, keyspace=astra_db_keyspace)
    collection = database.get_collection("financial_report")

    # Perform a similarity search
    query = "What are your thoughts on the stock market in 2025"
    results = collection.find(
        sort={"$vectorize": query},
        limit=5,
        projection={"$vectorize": True},
        include_similarity=True,
    )

    print(f"Vector search results for '{query}':")
    vector_texts = []
    for document in results:
        # Print retrieved documents
        print("    ", document)
        # Extract $vectorize field
        if "$vectorize" in document:
            vector_texts.append(document["$vectorize"])

    # Combine all $vectorize text as context
    context_text = "\n---\n".join(vector_texts)
    if not context_text.strip():
        print("No $vectorize fields found to summarize.")
        return

    # Create a prompt for OpenAI
    prompt = f"""
        You are an expert-level financial report analysis assistant. Your goal is to carefully examine the provided excerpts and deliver a highly accurate, insightful, and contextually rich summary of the key financial elements mentioned, such as earnings, revenue, guidance, margins, and other performance indicators. Before finalizing your response, critically evaluate your reasoning process and consider potential nuances or data points that might influence the interpretation.

        Context:
        {context_text}

        Instructions:
        1. Read the provided excerpts closely to identify all relevant financial metrics, trends, and forward-looking statements.
        2. Summarize these findings in a concise, yet comprehensive manner that captures the underlying financial health and strategy.
        3. Reflect on your interpretation: Are there any hidden assumptions or notable uncertainties? Consider how factors like market conditions, operational efficiency, or competitive landscape may affect the conclusions.
        4. Present the final summary in a clear, professional tone, suitable for a financial analyst's report.

        Please provide your final, self-reflective summary now.
    """

    # Use the new API interface for chat completion
    response = client_OpenAI.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    # Extract and print the summary
    summary = response.choices[0].message.content
    print("\nSummary:\n", summary)


if __name__ == "__main__":
    main()
