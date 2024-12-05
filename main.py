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

        # Initialize agents
        news_finder_agent = agents.news_finder()
        price_tracker_agent = agents.price_tracker()
        report_analyzer_agent = agents.report_analyzer()
        investment_advisor_agent = agents.investment_advisor()

        # Create tasks for each agent
        news_task = tasks.news_finder_task(news_finder_agent, self.company)
        price_task = tasks.price_tracker_task(price_tracker_agent, self.company)
        report_task = tasks.report_analyzer_task(report_analyzer_agent, self.company)
        recommendation_task = tasks.investment_advisor_task(investment_advisor_agent, self.company)


        # Create a Crew for task execution
        crew = Crew(
            agents=[
                news_finder_agent, 
                price_tracker_agent, 
                report_analyzer_agent, 
                investment_advisor_agent
            ],
            tasks=[
                news_task, 
                price_task, 
                report_task, 
                recommendation_task
            ],
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
    print("## Here is the final report: \n")
    print("########################\n")
    print(result)
