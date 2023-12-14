import logging
from typing import List
from os import linesep
import openai
import tiktoken
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from llmagents.language.protocols import ILanguageModel

logger = logging.getLogger('language_model')


class OpenAIChatModel(ILanguageModel):
    """Implements a language model based on OpenAI's chat model"""

    def __init__(self, api_key: str, model: str, temperature: float = 0.9, max_tokens: int = 1000, json: bool = False):
        """Initializes the OpenAI chat model with the specified model, temperature and max tokens"""
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.model: str = model
        self.encoding = tiktoken.encoding_for_model(self.model)
        self.openai = openai.AsyncOpenAI(api_key=api_key)
        self.json = json

    async def query(self, query: str, context: str, retry_count: int = 3) -> str:
        """Queries the language model with the specified query and returns the response"""
        messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": context},
                                                      {"role": "user", "content": query}]

        logger.info("Querying OpenAI")
        logger.info(f"{context}{linesep}{query}")
        # TODO: add context length management
        try:
            response = await self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={
                    "type": "json_object" if self.json else "text"},
            )
        except Exception as e:
            if retry_count > 0:
                return await self.query(query, context, retry_count - 1)
            raise e

        response_content = response.choices[0].message.content or ""

        logger.info("Response")
        logger.info(response_content)

        return response_content

    def encode(self, text: str) -> List[int]:
        """Encodes the specified text using the language model's encoding"""
        return self.encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Decodes the specified tokens using the language model's encoding"""
        return self.encoding.decode(tokens)
