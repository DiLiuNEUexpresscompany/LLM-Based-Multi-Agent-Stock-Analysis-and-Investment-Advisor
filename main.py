from tools.news_search import  NewsSearchTool
from agents.agent import ToolAgent, ToolRegistry

def main():
    # Setup
        # Register the tool
    registry = ToolRegistry()
    news_tool = NewsSearchTool()
    registry.register(news_tool)

    # Initialize the agent
    tool_agent = ToolAgent(registry)

    # Example of running a query:
    response = tool_agent.run("Search for news related to OpenAI")
    print(response)
if __name__ == "__main__":
    main()