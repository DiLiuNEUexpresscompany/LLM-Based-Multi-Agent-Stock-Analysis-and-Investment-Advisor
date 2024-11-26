from tools.news_search import  NewsSearchTool
from agents.agent import ToolAgent, ToolRegistry

def main():
    # Setup
    registry = ToolRegistry()
    registry.register(NewsSearchTool())
    
    # Run agent
    agent = ToolAgent(registry)
    result = agent.run("Search for Openai articles recently")
    print(result)

if __name__ == "__main__":
    main()