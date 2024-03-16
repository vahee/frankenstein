from typing import Dict, Any, Tuple
from datetime import datetime
import numpy as np

from agentopy import WithActionSpaceMixin, ActionResult, SharedStateKeys, IAgent, IAgentComponent, WithStateMixin, EntityInfo

from frankenstein.lib.language.protocols import IEmbeddingModel
from frankenstein.lib.db.protocols import IVectorDB


class Memory(WithActionSpaceMixin, WithStateMixin, IAgentComponent):
    """Implements a memory component"""

    def __init__(self, db: IVectorDB, embedding_model: IEmbeddingModel, memory_size: int = 3) -> None:
        """Initializes the memory component"""
        super().__init__()

        self.db: IVectorDB = db
        self.memory_size: int = memory_size
        self.embedding_model: IEmbeddingModel = embedding_model

    async def on_heartbeat(self, agent: IAgent) -> None:
        """Updates the component with the specified action and arguments"""
        memory_data: dict = {}

        result = agent.state.get_item(SharedStateKeys.AGENT_ACTION_RESULT)

        if not isinstance(result, ActionResult):
            return

        thoughts = agent.state.get_item(SharedStateKeys.AGENT_THOUGHTS)

        if isinstance(thoughts, dict):
            for k, v in thoughts.items():
                if isinstance(v, list):
                    v = ', '.join(v)
                if v is None:
                    v = ""
                assert isinstance(
                    v, str), f"At the moment memory only supports string values given {v} with type {type(v)}"
                memory_data[k] = v

        if isinstance(result.value, dict):
            for k, v in result.value.items():
                if isinstance(v, list):
                    v = ', '.join(v)
                if v is None:
                    v = ""
                assert isinstance(
                    v, str), f"At the moment memory only supports string values given {v} with type {type(v)}"
                memory_data[k] = v
        elif isinstance(result.value, str):
            memory_data["Action result"] = result.value

        if len(memory_data.keys()):

            memory_data["Memory type"] = 'Action was taken'

            memory_data["Action time"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S, %A")

            _, key = await self.add(memory_data)

            latest_memories_indices = await self.db.search(key, self.memory_size - 1)

            latest_memories = [await self.db.get(idx) for idx in latest_memories_indices]
            latest_memories.append(memory_data)

            for i, mem in enumerate(latest_memories):
                agent.state.set_item(
                    f"agent/components/{self.info().name}/{i}", mem)

        agent.state.remove_item(SharedStateKeys.AGENT_ACTION)
        agent.state.remove_item(SharedStateKeys.AGENT_ACTION_ARGS)
        agent.state.remove_item(SharedStateKeys.AGENT_ACTION_RESULT)
        agent.state.remove_item(SharedStateKeys.AGENT_THOUGHTS)

    async def add(self, value: Dict[str, Any] | str) -> Tuple[int, np.ndarray]:
        """Adds the specified value to the memory and returns the index and the key"""
        key: np.ndarray = await self._embed(value)

        idx: int = await self.db.add(key, value)

        return idx, key

    async def _embed(self, value: Any) -> np.ndarray:
        """Embeds the specified value"""
        text_values = []
        if isinstance(value, str):
            text_values.append(value)
        elif isinstance(value, dict):
            for key, value in value.items():
                if isinstance(value, str):
                    text_values.append(f"{key}: {value}")
        assert text_values, "Data item must have text values to be embeded for memory"
        return await self.embedding_model.embed(' '.join(text_values))

    def info(self) -> EntityInfo:
        """Returns the component info"""
        return EntityInfo(name="Memory", version="0.1.0", params={"memory_size": self.memory_size})
