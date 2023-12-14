from typing import List, Any
import numpy as np
from typing import Literal

from llmagents.db.protocols import IVectorDB


class InMemoryVectorDB(IVectorDB):
    """Implements an in-memory vector database"""

    def __init__(self, key_dim: int, affinity: Literal['cosine', 'dot'] = 'dot') -> None:
        """Initializes the in-memory database with the specified embedding model"""
        self.keys: np.ndarray = np.empty((0, key_dim))
        self.values: List[Any] = []
        self.key_dim: int = key_dim
        self.affinity = {
            'cosine': lambda x, y: np.dot(x, y) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y)),
            'dot': lambda x, y: np.dot(x, y)
        }[affinity]

    async def add(self, key: np.ndarray, value: Any) -> int:
        """Adds the specified key and value to the DB and returns the index"""
        self.keys = np.concatenate(
            (self.keys, np.array([key])))
        self.values.append(value)
        return len(self.values) - 1

    async def delete(self, idx: int) -> None:
        """Deletes the specified key from the DB"""
        del self.values[idx]
        self.keys = np.delete(self.keys, idx, axis=0)

    async def search(self, key: np.ndarray, k: int = 3) -> List[int]:
        """Returns top K values with most similar keys to the specified key"""
        similarities = self.affinity(self.keys, key)
        indices = np.argsort(similarities)[-k:]
        return indices.tolist()

    async def get(self, idx: int) -> Any:
        """Returns the value at the specified index"""
        return self.values[idx]

    async def clear(self) -> None:
        """Clears the DB"""
        self.values = []
        self.keys = np.empty((0, self.key_dim))
