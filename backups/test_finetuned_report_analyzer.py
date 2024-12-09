from backups.finetuned_report_analyst import FinetunedReportAnalyst

finetuned_report_analyzer_agent = FinetunedReportAnalyst()

question = "What are Tesla's projected revenue streams from energy storage and solar products beyond electric vehicle sales?"

response = finetuned_report_analyzer_agent.run(question)
print("=" * 100)
print(response)

# Save to file
with open('data/finetuned_report_analysis.txt', 'w', encoding='utf-8') as f:
    f.write(response)