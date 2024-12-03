from crews.analysis_agent import AnalysisAgent
from crews.task import Task
from crews.crew import Crew

# Create agents
market_researcher = AnalysisAgent(
    role="Market Research Analyst",
    goal="Collect and analyze market data",
    backstory="Experienced analyst with deep market insights",
    tools=[SearchTool(), BrowserTool()],
    verbose=True
)

financial_analyst = AnalysisAgent(
    role="Financial Analyst",
    goal="Perform in-depth financial analysis for the ",
    backstory="Expert in financial modeling and investment strategies",
    tools=[SearchTool()],
    verbose=True
)

# Create tasks
def print_result(result):
    print("Task completed with result:", result)

market_research_task = Task(
    description="Research market trends for tech companies",
    agent=market_researcher,
    expected_output="Comprehensive market trend report",
    callback=print_result
)

financial_analysis_task = Task(
    description="Analyze financial performance of major tech companies",
    agent=financial_analyst,
    expected_output="Detailed financial performance analysis",
    context=[market_research_task],
    callback=print_result
)

# Create and run crew
crew = Crew(
    agents=[market_researcher, financial_analyst],
    tasks=[market_research_task, financial_analysis_task],
    verbose=True
)

result = crew.kickoff()
print("\nFinal Crew Result:\n", result)

