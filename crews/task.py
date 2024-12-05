import uuid
from typing import Optional, List, Any, Callable

class Task:
    def __init__(
        self, 
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
        # Gather context from previous tasks
        context_info = ""
        for previous_task in self.context:
            if previous_task.result:
                context_info += f"Context from previous task: {previous_task.result}\n"
        
        # Combine context with task description
        full_description = context_info + "\n" + self.description
        
        # Execute task
        self.result = self.agent.execute_task(full_description)
        
        # Run callback if provided
        if self.callback:
            self.callback(self.result)
        
        return self.result