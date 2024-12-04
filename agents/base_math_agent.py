import os
import re
import json
from typing import Dict, Optional, List
from abc import ABC, abstractmethod
from openai import OpenAI
from dotenv import load_dotenv
from .tool_registry import ToolRegistry

import os
import re
import json
import time
import logging
from typing import Dict, Optional, List, Union
from abc import ABC, abstractmethod
from openai import OpenAI
from openai import APIError, InternalServerError, RateLimitError
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseMathAgent(ABC):
    """Base class for math-focused agents with robust error handling"""
    
    def __init__(self, registry: 'ToolRegistry', base_url: str = None, api_key: str = None,
                 max_retries: int = 3, retry_delay: float = 1.0):
        self.registry = registry
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        load_dotenv()
        
        # Initialize OpenAI client
        self.base_url = base_url or os.getenv("HUGGINGFACE_BASE_URLL")
        self.api_key = api_key or os.getenv("HUGGUNGFACE_ENDPOINT_API_KEY")
        
        if not self.base_url or not self.api_key:
            raise ValueError("API base URL and key must be provided either through constructor or environment variables")
            
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        # Default model parameters
        self.default_params = {
            "model": "tgi",
            "temperature": 0.7,
            "max_tokens": 150,
            "stream": False
        }

    def _stream_process(self, response) -> str:
        """Process streaming response from the model"""
        try:
            result = []
            for message in response:
                # 打印消息内容以便调试
                logger.debug(f"Stream message: {message}")
                
                # 检查不同的消息格式
                if hasattr(message, 'choices') and message.choices:
                    if hasattr(message.choices[0], 'delta'):
                        if hasattr(message.choices[0].delta, 'content'):
                            content = message.choices[0].delta.content
                            if content:
                                result.append(content)
                    elif hasattr(message.choices[0], 'message'):
                        if hasattr(message.choices[0].message, 'content'):
                            content = message.choices[0].message.content
                            if content:
                                result.append(content)
                
            final_result = ''.join(result)
            if not final_result:
                logger.warning("No content received from stream")
                return "No response content received from the model."
                
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}")
            return str(e)

    def _parse_tool_call(self, text: str) -> Dict:
        """Parse tool calls from model response"""
        pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.error("Failed to parse tool call JSON")
                return {}
        return {}

    def _make_api_call(self, messages: List[Dict], params: Dict):
        """Make API call with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    **params
                )
                
                # 检查响应是否为空
                if not response:
                    raise ValueError("Empty response received from API")
                
                # 打印响应内容以便调试
                logger.debug(f"API Response: {response}")
                
                # 如果是流式响应，直接返回
                if params.get('stream', False):
                    return response
                
                # 检查非流式响应的内容
                if not hasattr(response, 'choices') or not response.choices:
                    raise ValueError("Invalid response format: no choices found")
                
                if not hasattr(response.choices[0], 'message') or not response.choices[0].message.content:
                    raise ValueError("Invalid response format: no message content found")
                
                return response
                
            except (APIError, InternalServerError) as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
            except RateLimitError:
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Unexpected error in API call: {str(e)}")
                raise

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

    def run(self, user_input: str, stream: bool = False, **kwargs) -> str:
        """Execute the agent with robust error handling and response validation"""
        try:
            # Prepare parameters
            api_params = {k: v for k, v in kwargs.items() if k in self.default_params}
            tool_params = {k: v for k, v in kwargs.items() if k not in self.default_params}
            model_params = {**self.default_params, **api_params}
            model_params["stream"] = stream
            
            # Prepare system prompt and user input
            system_prompt = self.get_system_prompt()
            full_input = user_input
            if tool_params:
                full_input += f"\nParameters: {json.dumps(tool_params)}"
            
            # Log the request
            logger.debug(f"Making initial API call with input: {full_input}")
            
            # Initial API call
            try:
                response = self._make_api_call(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_input}
                    ],
                    params=model_params
                )
                
                # Process the response
                if stream:
                    initial_response = self._stream_process(response)
                else:
                    if not response.choices[0].message.content:
                        raise ValueError("Empty response content received")
                    initial_response = response.choices[0].message.content
                
                # Log the initial response
                logger.debug(f"Initial response: {initial_response}")
                
                if not initial_response:
                    return "No response received from the model. Please try again."
                
            except Exception as e:
                logger.error(f"Error in initial API call: {str(e)}")
                return f"Error in processing request: {str(e)}"
            
            # Parse tool call
            tool_call = self._parse_tool_call(initial_response)
            if not tool_call:
                logger.warning("Failed to parse tool call from the model's response.")
                return "Unable to parse tool call. Please rephrase your request."
        
            # Execute tool
            try:
                tool_name = tool_call.get("name")
                tool = self.registry.get_tool(tool_name)
                if not tool:
                    return f"The requested tool '{tool_name}' is not available."
                
                # Execute the tool with arguments
                tool_arguments = tool_call.get("arguments", {})
                tool_arguments.update(tool_params)
                arguments = self.process_tool_arguments(tool_name, tool_arguments)
                result = tool.execute(**arguments)
                formatted_result = self.format_tool_result(result)
                
                # Final API call with tool result
                final_response = self._make_api_call(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_input},
                        {"role": "assistant", "content": initial_response},
                        {"role": "user", "content": f"Tool result: {formatted_result}"}
                    ],
                    params=model_params
                )
                
                if stream:
                    return self._stream_process(final_response)
                else:
                    return final_response.choices[0].message.content
                
            except Exception as e:
                logger.error(f"Error in tool execution: {str(e)}")
                return f"Error in executing tool: {str(e)}"
                
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"

        
    def validate_numeric_input(self, value: Union[int, float, str], param_name: str, 
                             min_value: float = None, max_value: float = None) -> float:
        """Validate numeric inputs for mathematical operations"""
        try:
            num_value = float(value)
            if min_value is not None and num_value < min_value:
                raise ValueError(f"{param_name} must be greater than {min_value}")
            if max_value is not None and num_value > max_value:
                raise ValueError(f"{param_name} must be less than {max_value}")
            return num_value
        except (TypeError, ValueError):
            raise ValueError(f"{param_name} must be a valid number")

    def validate_list_input(self, value: any, param_name: str, 
                           min_length: int = None, max_length: int = None) -> List[float]:
        """Validate list inputs for mathematical operations"""
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{param_name} must be a list or tuple")
            
        try:
            numeric_list = [float(x) for x in value]
        except (TypeError, ValueError):
            raise ValueError(f"All elements in {param_name} must be valid numbers")
            
        if min_length is not None and len(numeric_list) < min_length:
            raise ValueError(f"{param_name} must have at least {min_length} elements")
        if max_length is not None and len(numeric_list) > max_length:
            raise ValueError(f"{param_name} must have no more than {max_length} elements")
            
        return numeric_list