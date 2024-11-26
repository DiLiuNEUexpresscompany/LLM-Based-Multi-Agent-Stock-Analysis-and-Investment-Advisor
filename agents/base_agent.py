import os
import re
import json
from typing import Dict, Optional
from abc import ABC, abstractmethod
from groq import Groq
from dotenv import load_dotenv
from .tool_registry import ToolRegistry

class BaseAgent(ABC):
    def __init__(self, registry: 'ToolRegistry'):
        self.registry = registry
        load_dotenv()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama3-groq-70b-8192-tool-use-preview"
    
    def _parse_tool_call(self, text: str) -> Dict:
        pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                return {}
        return {}
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for the specific agent type"""
        pass
    
    @abstractmethod
    def process_tool_arguments(self, tool_name: str, arguments: Dict) -> Dict:
        """Process and validate tool arguments before execution"""
        pass
    
    def format_tool_result(self, result: Dict) -> str:
        """Format tool execution result for final response generation"""
        return json.dumps(result)
    
    def run(self, user_input: str) -> str:
        system_prompt = self.get_system_prompt()
        # Get tool call from LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        tool_call = self._parse_tool_call(response.choices[0].message.content)
        if not tool_call:
            return "Unable to parse tool call. Please rephrase your request."

        tool_name = tool_call.get("name")
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return f"The requested tool '{tool_name}' is not available."

        try:
            arguments = self.process_tool_arguments(
                tool_name, 
                tool_call.get("arguments", {})
            )
            result = tool.execute(**arguments)
            formatted_result = self.format_tool_result(result)
            
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response.choices[0].message.content},
                    {"role": "user", "content": f"Tool result: {formatted_result}"}
                ]
            )
            return final_response.choices[0].message.content
            
        except Exception as e:
            return f"An error occurred while executing the tool: {e}"