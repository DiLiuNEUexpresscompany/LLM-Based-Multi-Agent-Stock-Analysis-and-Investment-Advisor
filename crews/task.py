# task.py
from typing import Optional, List, Any, Callable

class Task:
    def __init__(
        self, 
        agent_name: str,
        description: str, 
        agent: Any,
        expected_output: Optional[str] = None,
        context: Optional[List['Task']] = None,
        callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize a custom Task
        
        Args:
            description (str): Detailed description of the task
            agent (Agent): The agent responsible for executing the task
            expected_output (str, optional): Description of the expected output
            context (List[Task], optional): Previous tasks that provide context
            callback (Callable, optional): Function to call after task completion
        """
        self.agent_name = agent_name
        self.description = description
        self.agent = agent
        self.expected_output = expected_output or "Task output"
        self.context = context or []
        self.callback = callback
        self.result: Optional[str] = None
    
    def __str__(self):
        """String representation of the task"""
        return f"Task: {self.description[:50]}..."
    
    def execute(self, input_data = None) -> str:
        """
        Execute the task using the assigned agent
        
        Returns:
            str: Result of task execution
        """
        if input_data is not None:
            news_data = input_data.get('news_data', {})
            price_data = input_data.get('price_data', {})
            report_data = input_data.get('report_data', {})
            self.description = self.agent.generate_investment_analysis_prompt(news_data, price_data, report_data)

        result = self.agent.run(self.description)
        log_path = f"data/{self.agent_name}_output.md"
        with open(log_path, 'w', encoding='utf-8') as file:
            file.write(result)
        return result
        