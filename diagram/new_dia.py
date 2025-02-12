class NewsSearchTool:
    def search_news(self):
        return "News data retrieved."

class MarketDataTool:
    def fetch_market_data(self):
        return "Market data retrieved."

class ReportRetrievalTool:
    def retrieve_report(self):
        return "Report retrieved."

class InformationSynthesisTool:
    def synthesize_information(self, data):
        return f"Synthesis result based on: {data}"

class NewsSearchAgent:
    def __init__(self):
        self.tool = NewsSearchTool()

    def execute(self):
        return self.tool.search_news()

class MarketTrackingAgent:
    def __init__(self):
        self.tool = MarketDataTool()

    def execute(self):
        return self.tool.fetch_market_data()

class ReportAnalysisAgent:
    def __init__(self):
        self.tool = ReportRetrievalTool()

    def execute(self):
        return self.tool.retrieve_report()

class AdvisoryAgent:
    def __init__(self):
        self.tool = InformationSynthesisTool()

    def process_data(self, news_data, market_data, report_data):
        combined_data = f"{news_data}, {market_data}, {report_data}"
        return self.tool.synthesize_information(combined_data)

# Simulating the system
news_agent = NewsSearchAgent()
market_agent = MarketTrackingAgent()
report_agent = ReportAnalysisAgent()
advisory_agent = AdvisoryAgent()

news_data = news_agent.execute()
market_data = market_agent.execute()
report_data = report_agent.execute()

advisory_result = advisory_agent.process_data(news_data, market_data, report_data)
print(advisory_result)
