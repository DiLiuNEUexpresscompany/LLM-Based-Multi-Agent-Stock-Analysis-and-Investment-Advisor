from typing import List, Any 
from .task import Task

class Crew:
    def __init__(
        self, 
        agents: List[Any], 
        tasks: List[Task], 
        verbose: bool = False
    ):
        """
        Initialize a Crew of agents and tasks
        
        Args:
            agents (List[Agent]): List of agents in the crew
            tasks (List[Task]): List of tasks to be executed
            verbose (bool, optional): Whether to print detailed logs
        """
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose
    
    def kickoff(self) -> str:
        """
        Execute all tasks in sequence
        
        Returns:
            str: Final aggregated result of all tasks
        """
        if self.verbose:
            print("Crew starting task execution...")
        
        final_results = []
        for task in self.tasks:
            if self.verbose:
                print(f"\nExecuting: {task}")
            
            # Execute the task
            result = task.execute()
            final_results.append(result)
        
        # Combine and return final results
        return "\n\n".join(final_results)