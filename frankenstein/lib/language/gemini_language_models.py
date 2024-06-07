import logging
from typing import List
from os import linesep
import google.generativeai as genai
import tiktoken
from frankenstein.lib.language.protocols import ILanguageModel

logger = logging.getLogger('language_model')


class GeminiAIChatModel(ILanguageModel):
    """Implements a language model based on Gemini's chat model"""

    def __init__(self, api_key: str, model: str):
        """Initializes the Gemini chat model with the specified model"""
        self.model = genai.GenerativeModel(model)
        genai.configure(api_key=api_key)
        self.encoding = tiktoken.get_encoding('cl100k_base')

    async def query(self, query: str, context: str, retry_count: int = 3) -> str:
        """Queries the language model with the specified query and returns the response"""
        prompt = f"Context: {context}{linesep}Query: {query}"

        logger.info("Querying Gemini")
        logger.info(f"{context}{linesep}{query}")
        # TODO: add context length management
        try:
            response = await self.model.generate_content_async(prompt)
        except Exception as e:
            if retry_count > 0:
                return await self.query(query, context, retry_count - 1)
            raise e

        response_content = response.text

        logger.info("Response")
        logger.info(response_content)

        return response_content

    def encode(self, text: str) -> List[int]:
        """Encodes the specified text using the language model's encoding"""
        return self.encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Decodes the specified tokens using the language model's encoding"""
        return self.encoding.decode(tokens)