import os
import re
import json
from typing import Dict
from groq import Groq
from dotenv import load_dotenv
from tools.base_tool import BaseTool

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        self.tools[tool.name()] = tool
    
    def get_tool(self, name: str) -> BaseTool:
        return self.tools.get(name)
    
    def get_tools_info(self) -> str:
        tools_info = [
            {
                "name": tool.name(),
                "description": tool.description()
            }
            for tool in self.tools.values()
        ]
        return json.dumps(tools_info)

class ToolAgent:
    def __init__(self, registry: ToolRegistry):
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
    
    def run(self, user_input: str) -> str:
        tools_info = self.registry.get_tools_info()
        system_prompt = f"""
        Available tools:
        {tools_info}
        
        Call tools using format:
        <tool_call>{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}</tool_call>
        """
        
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
            arguments = tool_call.get("arguments", {})

            # 参数映射
            if tool_name == "search_news":
                arguments["query"] = arguments.pop("keyword", arguments.get("query"))

            result = tool.execute(**arguments)
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response.choices[0].message.content},
                    {"role": "user", "content": f"Tool result: {json.dumps(result)}"}
                ]
            )
            return final_response.choices[0].message.content
        except Exception as e:
            return f"An error occurred while executing the tool: {e}"

        
        return response.choices[0].message.content