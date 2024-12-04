import json
from typing import Dict, List,Optional
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
    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    def remove_tool(self, name: str) -> None:
        """Remove a tool from the registry"""
        if name in self.tools:
            del self.tools[name]
    
    def clear(self) -> None:
        """Remove all tools from the registry"""
        self.tools.clear()
    
    def list_tool_names(self) -> List[str]:
        """Get a list of all registered tool names"""
        return list(self.tools.keys())
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool exists in the registry"""
        return name in self.tools