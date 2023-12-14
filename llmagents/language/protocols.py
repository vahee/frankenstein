from typing import Optional, List, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class ILanguageModel(Protocol):
    """Interface for language models"""

    async def query(self, query: str, context: Optional[str]) -> str:
        """Queries the language model with the specified query and context"""
        ...

    def encode(self, text: str) -> List[int]:
        """Encodes the specified text into a list of tokens"""
        ...

    def decode(self, tokens: List[int]) -> str:
        """Decodes the specified tokens into a string"""
        ...


@runtime_checkable
class IEmbeddingModel(Protocol):
    """Interface for embedding models"""
    async def embed(self, text: str) -> np.ndarray:
        """Returns an embedding for the input list"""
        ...

    def dim(self) -> int:
        """Returns the dimension of the embeddings"""
        ...
