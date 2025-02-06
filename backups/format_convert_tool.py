import os
import json
from typing import Dict, Any, Union
from dotenv import load_dotenv
from groq import Groq
from tools.base_tool import BaseTool

class FormatConvertTool(BaseTool):
    """Tool to convert text using an LLM (Groq API)"""
    
    def __init__(self, registry=None):
        load_dotenv()
        self.registry = registry
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
    
    def name(self) -> str:
        return "llm_text_converter"
    
    def description(self) -> str:
        return """Convert text using LLM (Groq API).
        Arguments:
        - text (required): Input text to be converted
        - output_format (required): Desired output format (json, dict, etc.)
        - conversion_instructions (optional): Specific conversion details"""
    
    def _construct_prompt(self, text: str, output_format: str, 
                          conversion_instructions: str = None) -> str:
        """
        Construct a prompt for the LLM to convert text
        
        :param text: Input text to convert
        :param output_format: Desired output format
        :param conversion_instructions: Additional specific instructions
        :return: Constructed prompt
        """
        prompt = f"""You are an expert text converter. Your task is to transform the following input text into a structured {output_format} format.

            Input Text:
            {text}

            Conversion Requirements:
            1. Convert the text into a valid {output_format} format
            2. Preserve the original meaning and key information
            3. Use clear, concise keys
            4. Handle any potential parsing challenges

            {f"Additional Specific Instructions: {conversion_instructions}" if conversion_instructions else ""}

            Please provide ONLY the converted {output_format} output. Do not include any explanatory text or markdown code blocks.
        """
        
        return prompt
    
    def execute(self, text: str = None, output_format: str = 'json', 
                conversion_instructions: str = None, **kwargs) -> Union[Dict, str]:
        """
        Convert text using LLM
        
        :param text: Input text to convert
        :param output_format: Desired output format
        :param conversion_instructions: Specific conversion details
        :return: Converted output or error message
        """
        if not text:
            return {"error": "No text provided for conversion"}
        
        try:
            # Construct the prompt
            prompt = self._construct_prompt(
                text, 
                output_format, 
                conversion_instructions
            )
            
            # Make API call to Groq
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant specialized in text conversion."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"} if output_format == 'json' else None
            )
            
            # Extract and parse the response
            response_text = chat_completion.choices[0].message.content.strip()
            
            # Try to parse the response based on output format
            if output_format == 'json':
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    return {"error": f"Failed to parse JSON: {response_text}"}
            
            elif output_format == 'dict':
                try:
                    return eval(response_text)
                except (SyntaxError, ValueError):
                    return {"error": f"Failed to parse dictionary: {response_text}"}
            
            else:
                return {"error": f"Unsupported output format: {output_format}"}
        
        except Exception as e:
            return {"error": f"Conversion error: {str(e)}"}