import os
import requests
import json
from typing import Optional, Dict, Any, List

class OllamaClient:
    def __init__(self):
        """
        Initialize the Ollama client.
        
        Args:
            model_name (str): Name of the model to use (default: "llama2")
        """
        self.model_name = os.getenv('LLM_MODEL')
        self.base_url = os.getenv('OLLAMA_BASE_URL')
        
    def chat_completion(self, 
                       prompt: str, 
                       system_prompt: str = "You are a helpful assistant.",
                       json_response: bool = False,
                       temperature: float = 0.0) -> str:
        """
        Send a chat completion request to Ollama.
        
        Args:
            prompt (str): The user prompt
            system_prompt (str): The system prompt
            json_response (bool): Whether to expect JSON response
            temperature (float): Temperature for response generation
            
        Returns:
            str: The model's response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        if json_response:
            payload["format"] = "json"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['message']['content']
            
            if json_response:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract JSON from the response
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(1))
                    raise ValueError("Failed to parse JSON response from Ollama")
            
            return content
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error communicating with Ollama: {str(e)}")
            
    def list_models(self) -> List[str]:
        """
        List available models in Ollama.
        
        Returns:
            List[str]: List of available model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return [model['name'] for model in response.json()['models']]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error listing Ollama models: {str(e)}")
            
    def pull_model(self, model_name: str) -> None:
        """
        Pull a model from Ollama.
        
        Args:
            model_name (str): Name of the model to pull
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name}
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error pulling Ollama model: {str(e)}") 