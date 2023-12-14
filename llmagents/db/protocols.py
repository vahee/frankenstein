from typing import List, Any, Protocol
import numpy as np


class IVectorDB(Protocol):
    async def add(self, key: np.ndarray, value: Any) -> int:
        """Adds the specified key and value to the DB"""
        ...

    async def delete(self, idx: int) -> None:
        """Deletes the specified key from the DB"""
        ...

    async def search(self, key: np.ndarray, k: int) -> List[int]:
        """Returns top K values with most similar keys to the specified key"""
        ...

    async def get(self, idx: int) -> Any:
        """Returns the value at the specified index"""
        ...

    async def clear(self) -> None:
        """Clears the DB"""
        ...
