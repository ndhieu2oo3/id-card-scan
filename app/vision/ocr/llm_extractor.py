"""LLM integration for structured information extraction.

This module implements a lightweight LLM client that sends OpenAI-compatible
chat completion requests to an external LLM HTTP endpoint configured by the
`LLM_API_URL` environment variable. The client intentionally avoids trying to
start or probe a local model server so the application can rely on an
already-served external LLM (TGI, hosted service, etc.).
"""

import requests
import json
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


class LLMClient:
    """Client for querying an external OpenAI-compatible LLM HTTP API.

    The client expects the external API to support the OpenAI Chat Completions
    interface (POST to `.../v1/chat/completions` with `messages` and return
    a `choices` array). The endpoint URL is taken from `LLM_API_URL`.
    """

    def __init__(self,
                 api_url: str = None,
                 model: str = None,
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 512):
        self.api_url = api_url or os.environ.get('LLM_API_URL', 'https://qwen25-7b-instruct.ghtklab.com/v1/chat/completions')
        self.model = model or os.environ.get('LLM_MODEL', 'Qwen2.5-7B-Instruct')
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
    def query(self, user_message: str, system_message: str = "You are a helpful assistant for information extraction.", timeout: int = 60) -> Optional[str]:
        """
        Query vLLM API with a prompt.
        
        Args:
            user_message: User prompt
            system_message: System message for context
            timeout: Request timeout in seconds
            
        Returns:
            LLM response or None if failed
        """
        try:
            headers = {'Content-Type': 'application/json'}
            payload = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': system_message},
                    {'role': 'user', 'content': user_message}
                ],
                'temperature': self.temperature,
                'top_p': self.top_p,
                'max_tokens': self.max_tokens
            }

            response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()

            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                # OpenAI-compatible response
                choice = result['choices'][0]
                # Chat-style response
                if isinstance(choice.get('message'), dict):
                    return choice['message'].get('content')
                # Completion-style response
                if 'text' in choice:
                    return choice.get('text')

            return None

        except requests.exceptions.Timeout:
            print(f"LLM query timeout after {timeout}s")
            return None
        except requests.exceptions.ConnectionError:
            print(f"Failed to connect to LLM API at {self.api_url}. Is it configured and reachable?")
            return None
        except Exception as e:
            print(f"LLM query error: {e}")
            return None
    
    def extract_structured_info(self, context: str, fields: List[str], timeout: int = 60) -> Optional[Dict]:
        """
        Extract structured information from context.
        
        Args:
            context: Text context from OCR
            fields: List of fields to extract
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with extracted fields or None if failed
        """
        if not fields:
            return {}

        fields_str = '\n'.join([f"- {field}" for field in fields])
        user_message = (
            f"QUERY: Từ đoạn text dưới đây, trích xuất trường thông tin sau: {fields_str}\n\n"
            f"INSTRUCTION: Trả về dạng JSON với key là tên trường thông tin cần trích xuất, không giải thích gì thêm.\n"
            f"CONTEXT: {context}"
        )

        response = self.query(user_message, timeout=timeout)
        if not response:
            return None

        try:
            # Clean markdown code fences if present
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {e}")
            return None

def parallel_field_extraction_with_descriptions(context: str,
                                               info_dict: Dict[str, str],
                                               batch_size: int = 3,
                                               timeout: int = 30) -> Dict[str, str]:
    """
    Extract fields in parallel using descriptions from info_dict.
    
    Args:
        context: Text context from OCR
        info_dict: Dict mapping field_key -> field_description (Vietnamese)
        batch_size: Number of parallel requests
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with extracted field values keyed by field_key
    """
    if not info_dict:
        return {}
    
    client = LLMClient()
    results = {}
    
    def extract_field_with_description(field_key: str, description: str) -> tuple:
        """Extract single field using its description."""
        user_message = f"QUERY: Từ đoạn text dưới đây, trích xuất trường thông tin sau: {description}\n\nINSTRUCTION: Trả về dạng JSON với key là '{field_key}', không giải thích gì thêm.\nCONTEXT: {context}"
        response = client.query(user_message, timeout=timeout)
        
        # Parse JSON response and extract the field value
        if response:
            try:
                response_str = response.strip()
                # Handle markdown code blocks if present
                if response_str.startswith('```json'):
                    response_str = response_str[7:]
                if response_str.startswith('```'):
                    response_str = response_str[3:]
                if response_str.endswith('```'):
                    response_str = response_str[:-3]
                response_str = response_str.strip()
                
                # Parse JSON and extract field value
                parsed = json.loads(response_str)
                value = parsed.get(field_key, "")
                return field_key, value
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                print(f"Failed to parse response for field {field_key}: {e}")
                return field_key, response or ""
        
        return field_key, ""
    
    # Use ThreadPoolExecutor for parallel extraction
    field_items = list(info_dict.items())
    with ThreadPoolExecutor(max_workers=min(batch_size, len(field_items))) as executor:
        futures = {executor.submit(extract_field_with_description, key, desc): key 
                  for key, desc in field_items}
        
        for future in as_completed(futures):
            try:
                field_key, value = future.result()
                results[field_key] = value or ""
            except Exception as e:
                print(f"Error extracting field: {e}")
                results[futures[future]] = ""
    
    return results
