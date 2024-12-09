from openai import OpenAI
from dotenv import load_dotenv
import os
from astrapy.client import DataAPIClient
from textwrap import dedent
def create_test_prompt(data_row):
    # Format the question and context into the desired structure
    prompt = dedent(f"""
        You are an expert-level financial report analysis assistant. Your goal is to carefully examine the provided excerpts and deliver a highly accurate, insightful, and contextually rich analysis of the key financial elements mentioned, such as earnings, revenue, guidance, margins, and other performance indicators. 
                            
            Question:
            {data_row["question"]}

            Context:
            {data_row["context"]}

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
def main():
    # Load the environment variables
    load_dotenv()

    # Fetch required environment variables
    hf_api_key = os.getenv("HUGGUNGFACE_ENDPOINT_API_KEY")
    astra_db_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    astra_db_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    astra_db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

    # Ensure all required variables are present
    required_vars = {
        "HUGGUNGFACE_ENDPOINT_API_KEY": hf_api_key,
        "ASTRA_DB_APPLICATION_TOKEN": astra_db_token,
        "ASTRA_DB_API_ENDPOINT": astra_db_endpoint,
        "ASTRA_DB_KEYSPACE": astra_db_keyspace,
    }

    for var_name, value in required_vars.items():
        if not value:
            raise ValueError(f"Missing required environment variable: {var_name}")

    # Connect to the Hugging Face TGI client
    client_hf = OpenAI(
        base_url="https://b4hv8vle4pryfmy8.us-east-1.aws.endpoints.huggingface.cloud/v1/",
        api_key=hf_api_key
    )

    # Connect to the Astra database
    client = DataAPIClient(astra_db_token)
    database = client.get_database(astra_db_endpoint, keyspace=astra_db_keyspace)
    collection = database.get_collection("financial_report")

    # Perform a similarity search
    question = "What percentage of global electricity usage does NVIDIA aim to match with renewable energy by the end of fiscal year 2025?"
    results = collection.find(
        sort={"$vectorize": question},
        limit=5,
        projection={"$vectorize": True},
        include_similarity=True,
    )

    print(f"Vector search results for '{question}':")
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

    # Prepare data row for prompt creation
    data_row = {
        "question": question,
        "context": context_text,
    }

    # Create prompt using the provided template
    messages = create_test_prompt(data_row)

    # Use the Hugging Face TGI model to generate a response
    chat_completion = client_hf.chat.completions.create(
        model="tgi",
        messages=messages,
        temperature=0.1,
        max_tokens=512,
        stream=True,
    )

    # Stream and print the generated response
    print("\nGenerated Response:\n")
    for message in chat_completion:
        print(message.choices[0].delta.content, end="")


if __name__ == "__main__":
    main()