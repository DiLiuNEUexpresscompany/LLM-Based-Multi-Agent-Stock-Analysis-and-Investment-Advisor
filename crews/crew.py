from typing import List, Any 

class Crew:
    def __init__(
        self, 
        tasks: dict,
        verbose: bool = False
    ):
        """
        Initialize a Crew of agents and tasks
        
        Args:
            agents (List[Agent]): List of agents in the crew
            tasks (List[Task]): List of tasks to be executed
            verbose (bool, optional): Whether to print detailed logs
        """
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