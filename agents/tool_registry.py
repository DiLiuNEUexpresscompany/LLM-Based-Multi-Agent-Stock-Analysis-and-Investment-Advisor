import json
from typing import Dict, Optional
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