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
    
    def execute(self) -> str:
        """
        Execute the task using the assigned agent
        
        Returns:
            str: Result of task execution
        """
        result = self.agent.run(self.description)
        log_path = f"data/{self.agent_name}_output.txt"
        with open(log_path, 'w') as file:
            file.write(result)
        return result
        