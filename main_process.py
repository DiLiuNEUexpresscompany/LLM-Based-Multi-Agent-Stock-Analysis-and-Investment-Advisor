from crews.crew import Crew
from textwrap import dedent

from stock_analysis_agents import StockAnalysisAgents
from stock_analysis_tasks import StockAnalysisTasks

from dotenv import load_dotenv

# Load environment variables from a file
dotenv_path = '.env'
load_dotenv(dotenv_path)

class FinancialCrew:
    def __init__(self, company):
        """Initialize with the company name to be analyzed."""
        self.company = company

    def run(self):
        """Run the analysis by initializing agents and tasks."""
        # Initialize the agents and tasks
        agents = StockAnalysisAgents()
        tasks = StockAnalysisTasks()

        # Assign specific agents to respective tasks
        research_analyst_agent = agents.research_analyst()
        financial_analyst_agent = agents.financial_analyst()
        investment_advisor_agent = agents.investment_advisor()

        # Assign tasks for each agent
        research_task = tasks.research(research_analyst_agent, self.company)
        financial_task = tasks.financial_analysis(financial_analyst_agent, self.company)
        filings_task = tasks.filings_analysis(financial_analyst_agent, self.company)
        recommend_task = tasks.recommend(investment_advisor_agent, self.company)

        # Create a Crew for task execution
        crew = Crew(
            agents=[research_analyst_agent, financial_analyst_agent, investment_advisor_agent],
            tasks=[research_task, financial_task, filings_task, recommend_task],
            verbose=True
        )

        # Execute the tasks and return the result
        result = crew.kickoff()
        return result

if __name__ == "__main__":
    # Welcome message for users
    print("## Welcome to the Financial Analysis Team")
    print('-------------------------------')

    # Prompt for company name
    company = input(
        dedent("""
            What company would you like to analyze?
        """)
    )
  
    # Run the analysis
    financial_crew = FinancialCrew(company)
    result = financial_crew.run()

    # Display the analysis report
    print("\n\n########################")
    print("## Here is the report")
    print("########################\n")
    print(result)
