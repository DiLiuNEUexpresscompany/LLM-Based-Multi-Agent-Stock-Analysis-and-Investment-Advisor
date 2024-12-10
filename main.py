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
        news_search_agent = agents.news_searcher()
        price_track_agent = agents.price_tracker()
        report_analysis_agent = agents.report_analyst()
        investment_advice_agent = agents.investment_advisor()

        # Create tasks for each agent
        news_task = tasks.news_search_task(news_search_agent, self.company)
        price_task = tasks.price_track_task(price_track_agent, self.company)
        report_task = tasks.report_analysis_task(report_analysis_agent, self.company)
        recommendation_task = tasks.investment_advice_task(investment_advice_agent, self.company)

        result = {}

        news_search_result = news_task.execute()
        result.append({"news_data", news_search_result})

        price_track_result = price_task.execute()
        result.append({"price_data", price_track_result})

        report_analyst_result = report_task.execute()
        result.append({"report_data", report_analyst_result})

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
    with open('data/financial_report.txt', 'a') as f:
        f.write(f"{result}\n")
