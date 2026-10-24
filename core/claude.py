import os
from anthropic import Anthropic
from anthropic.types import Message


class Claude:
    def __init__(self, model: str):
        api_endpoint = os.getenv("API_ENDPOINT")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_endpoint:
            self.client = Anthropic(base_url=api_endpoint, api_key=api_key)
        else:
            self.client = Anthropic()
        self.model = model

    def add_user_message(self, messages: list, message):
        user_message = {
            "role": "user",
            "content": message.content
            if isinstance(message, Message)
            else message,
        }
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message):
        assistant_message = {
            "role": "assistant",
            "content": message.content
            if isinstance(message, Message)
            else message,
        }
        messages.append(assistant_message)

    def text_from_message(self, message: Message):
        # Handle standard Anthropic format
        if message.content:
            texts = []
            for block in message.content:
                if block.type == "text":
                    texts.append(block.text)
            return "\n".join(texts)
        
        # Handle OpenAI format (from proxy)
        if hasattr(message, 'choices') and message.choices:
            content = message.choices[0]['message']['content']
            return content if content else ""
        
        return ""

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> Message:
        params = {
            "model": self.model,
            "max_tokens": 8000,
            "messages": messages,
        }

        if temperature != 1.0:
            params["temperature"] = temperature

        if stop_sequences:
            params["stop_sequences"] = stop_sequences

        if thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        if tools:
            params["tools"] = tools

        if system:
            params["system"] = system

        message = self.client.messages.create(**params)
        return message
