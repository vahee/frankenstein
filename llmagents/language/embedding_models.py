import openai
import logging
import numpy as np
from typing import Literal
from sentence_transformers import SentenceTransformer

from llmagents.language.protocols import IEmbeddingModel

logger = logging.getLogger('language_model')


class OpenAIEmbeddingModel(IEmbeddingModel):
    """Implements an embedding model that uses OpenAI's GPT-2 model"""

    def __init__(self, api_key: str, model: str):
        """Initializes the OpenAI embedding model with the specified model"""
        self.model: str = model
        self.openai = openai.AsyncOpenAI(api_key=api_key)

    async def embed(self, text: str, retry_count: int = 3) -> np.ndarray:
        """Embeds the input list"""
        try:
            response = await self.openai.embeddings.create(model=self.model, input=[text])
        except Exception as e:
            if retry_count > 0:
                return await self.embed(text, retry_count - 1)
            raise e
        embedding = response.data[0].embedding  # type: ignore
        return embedding  # type: ignore

    def dim(self) -> int:
        """Returns the dimension of the embeddings"""
        return 1536


class SentenceTransformerEmbeddingModel(IEmbeddingModel):
    """Implements an embedding model that uses the Sentence Transformer model"""

    def __init__(self, model_name: str, access_token: str, device: Literal['cuda', 'cpu'] = 'cpu'):
        """Initializes the Sentence Transformer embedding model with the specified model"""
        self.model = SentenceTransformer(
            model_name, token=access_token, device=device)

    async def embed(self, text: str) -> np.ndarray:
        """Embeds the input list"""
        return self.model.encode(text, convert_to_numpy=True)  # type: ignore

    def dim(self) -> int:
        """Returns the dimension of the embeddings"""
        return 384
