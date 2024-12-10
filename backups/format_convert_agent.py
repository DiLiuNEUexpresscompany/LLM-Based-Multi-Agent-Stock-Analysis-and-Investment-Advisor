from typing import Any
from agents.tool_registry import ToolRegistry

class FormatConverter:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    def run(self, text: str, output_format: str = 'json', 
            conversion_instructions: str = None) -> Any:
        """
        Run format conversion process
        
        :param text: Input text to convert
        :param output_format: Desired output format
        :param conversion_instructions: Specific conversion details
        :return: Converted output
        """
        try:
            # Get the LLM text converter tool
            converter_tool = self.registry.get_tool('llm_text_converter')
            
            # Execute the conversion
            result = converter_tool.execute(
                text=text,
                output_format=output_format,
                conversion_instructions=conversion_instructions
            )
            
            return result
        
        except Exception as e:
            return {
                "error": f"Format conversion failed: {str(e)}",
                "details": str(e)
            }