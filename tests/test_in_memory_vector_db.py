import pytest
import numpy as np

from frankenstein.lib.db.in_memory_vector_db import InMemoryVectorDB


@pytest.mark.asyncio
async def test_dot_product_affinity():
    db = InMemoryVectorDB(3)

    await db.add(np.array([1, 2, 3]), 1)
    await db.add(np.array([3, 2, 1]), 2)
    await db.add(np.array([2, 1, 3]), 3)

    idx_1 = await db.search(np.array([3, 1, 1]), 1)
    val_1 = await db.get(idx_1[0])
    assert val_1 == 2

    await db.add(np.array([6, 4, 2]), 4)

    idx_2 = await db.search(np.array([1, 2, 3]), 1)
    val_2 = await db.get(idx_2[0])
    assert val_2 == 4


@pytest.mark.asyncio
async def test_consine_affinity():
    db = InMemoryVectorDB(3, 'cosine')

    await db.add(np.array([1, 2, 3]), 1)
    await db.add(np.array([3, 2, 1]), 2)
    await db.add(np.array([2, 1, 3]), 3)

    idx_1 = await db.search(np.array([3, 1, 1]), 1)
    val_1 = await db.get(idx_1[0])
    assert val_1 == 2

    await db.add(np.array([6, 4, 2]), 4)

    idx_2 = await db.search(np.array([1, 2, 3]), 1)
    val_2 = await db.get(idx_2[0])
    assert val_2 == 1
