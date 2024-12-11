# LLM-Based Multi-Agent Stock Analysis and Investment Advisor

This project leverages Large Language Models (LLMs) and a multi-agent framework to analyze stock prices, gather relevant news, and generate comprehensive financial investment reports for specific companies.

---

## Project Structure

```plaintext
.
├── README.md
├── agents
│   ├── base_agent.py          # Base class for agents
│   ├── investment_advisor.py  # Advises on investment decisions
│   ├── news_searcher.py       # Searches for relevant financial news
│   ├── price_tracker.py       # Tracks stock price data
│   ├── report_analyst.py      # Analyzes and generates financial reports
│   └── tool_registry.py       # Registers and manages tools for agents
├── crews
│   ├── crew.py                # Defines agent crews
│   └── task.py                # Task management for crews
├── main.py                    # Main entry point of the application
├── requirements.txt           # List of Python dependencies
├── stock_analysis_agents.py   # High-level setup for agents
├── stock_analysis_tasks.py    # Task definitions for stock analysis
├── tests
│   ├── __init__.py
│   ├── test_analysis_agent.py
│   ├── test_investment_advisor.py
│   ├── test_news_searcher.py
│   ├── test_price_tracker.py
│   ├── test_report_analyst.py
│   └── test_tools.py
└── tools
    ├── base_tool.py           # Base class for tools
    ├── data_analysis_tool.py  # Analyzes stock data
    ├── news_search_tool.py    # Fetches financial news articles
    ├── report_analysis_tool.py# Processes and analyzes financial reports
    ├── report_retrieval_tool.py # Retrieves financial reports
    └── stock_price_tool.py    # Fetches stock price data
```

---

## Key Features

- **Multi-Agent Design**: Specialized agents collaborate to perform specific tasks such as tracking stock prices, fetching news, and advising investments.
- **Tool Integration**: Tools provide modular and reusable functionalities for the agents.
- **Task Management**: A `crew` system ensures efficient task execution and delegation.
- **Automated Reports**: Generates comprehensive financial investment reports based on the analysis.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Pip (Python package manager)
- [OpenAI API Key](https://platform.openai.com/signup) (if LLM integration is required)

### Installation

1. Clone the repository:

    ```bash
    git clone git@github.com:DiLiuNEUexpresscompany/LLM-Based-Multi-Agent-Stock-Analysis-and-Investment-Advisor.git
    cd LLM-Based-Multi-Agent-Stock-Analysis-and-Investment-Advisor
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your API keys (if applicable) in a `.env` file:

    ```plaintext
    GROQ_API_KEY="Your GROQ API key"
    NEWS_API_KEY="Your News API key"
    STOCK_API_KEY="Your Stock API key"
    ASTRA_DB_API_ENDPOINT="Your Astra DB API endpoint"
    ASTRA_DB_APPLICATION_TOKEN="Your Astra DB application token"
    ASTRA_DB_KEYSPACE="Your Astra DB keyspace"
    OPENAI_API_KEY="Your OpenAI API key"
    HUGGUNGFACE_ENDPOINT_API_KEY="Your Huggingface Endpoint API key"
    HUGGINGFACE_BASE_URL="Your Huggingface Base Url"
    ```

### Running the Application

Execute the main script to start the analysis process:

```bash
streamlit run main.py
```

---

## Directory Details

### `agents`
This directory contains the implementation of different agents responsible for specific tasks:
- **`base_agent.py`**: Abstract base class for all agents.
- **`investment_advisor.py`**: Suggests investment strategies based on data.
- **`news_searcher.py`**: Retrieves recent financial news for the target company.
- **`price_tracker.py`**: Monitors historical and real-time stock prices.
- **`report_analyst.py`**: Combines outputs to create a financial report.
- **`tool_registry.py`**: Manages the tools available to the agents.

### `crews`
Manages agent collaboration and task delegation.

- **`crew.py`**: Defines a group of agents working together.
- **`task.py`**: Handles task assignments and execution.

### `tools`
Encapsulates tools used by the agents to perform specific operations:
- `data_analysis_tool.py`
- `news_search_tool.py`
- `report_analysis_tool.py`
- `report_retrieval_tool.py`
- `stock_price_tool.py`

### `tests`
Unit tests for individual components of the system. Run tests with:

```bash
python test_news_searcher.py
python test_price_tracker.py
python test_report_analyst.py
python test_investment_advisor.py
```
---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [OpenAI](https://openai.com) for providing state-of-the-art LLMs.
- Financial data APIs and libraries.

---

## Contact

For questions or suggestions, please contact:

liudix7@gmail.com

chongchen1999@gmail.com
