from typing import Dict, Any, Optional, Callable, Awaitable
import logging
import asyncio as aio
import time

from agentopy import WithActionSpaceMixin, IAgentComponent, IAgent, IAction, IState, EntityInfo

from frankenstein.lib.networking.communication import IMessaging

logger = logging.getLogger('[Component][RemoteControl]')


class RemoteControl(WithActionSpaceMixin, IAgentComponent):
    """Implements a remote control component that allows to control the agent remotely."""

    def __init__(self, messaging: IMessaging, subscription_update_rate_ms: int = 1000):
        super().__init__()
        self._messaging: IMessaging = messaging
        self._agent_state_subscription: Dict[str, Any] | None = None
        self._agent: IAgent | None = None
        self._subscription_update_rate_ms: int = subscription_update_rate_ms        
        
    async def tick(self) -> None:
        if not self._agent:
            return
        
        if not await self._messaging.is_connected():
            self._agent_state_subscription = None
            await aio.sleep(1)
        else:

            if self._agent_state_subscription is not None:
                if time.time() - self._agent_state_subscription.get("ts", 0) > self._subscription_update_rate_ms / 1000:
                    await self.get_agent_state(self._agent, self._agent_state_subscription.get("data"))
                    self._agent_state_subscription["ts"] = time.time()

            message = await self._messaging.get(1)
            if message is not None:
                await self.process_message(self._agent, message)
                
    async def on_agent_heartbeat(self, agent: IAgent) -> None:
        
        if not self._agent:
            self._agent = agent

    async def process_message(self, agent: IAgent, message: Dict[str, Any]) -> None:
        """Processes the message"""
        if not isinstance(message, Dict):
            await self._messaging.send_message({"error": "Invalid message"})
            return
        
        command: Callable[..., Awaitable] = {
            "get_agent_state": self.get_agent_state,
            "subscribe_agent_state": self.subscribe_agent_state,
            "unsubscribe_agent_state": self.unsubscribe_agent_state,
            "get_agent_info": self.get_agent_info,
            "force_action": self.force_action,
            "authenticate": self.authenticate,
        }.get(message.get("command", None), self.invalid_command)
        
        await command(agent, message.get("data"))

    async def invalid_command(self, _agent: IAgent, _data: Dict) -> None:
        """Does nothing"""
        await self._messaging.send_message({"error": "Invalid command"})

    async def subscribe_agent_state(self, _agent: IAgent, data: Optional[Dict[str, Any]]) -> None:
        """Subscribes to the agent's state"""
        self._agent_state_subscription = {
            "data": data,
            "ts": time.time()
        }

    async def unsubscribe_agent_state(self, _agent: IAgent, _data: Optional[Dict[str, Any]]) -> None:
        """Unsubscribes from the agent's state"""
        self._agent_state_subscription = None

    async def get_agent_state(self, agent: IAgent, _data: Optional[Dict[str, Any]]) -> None:
        """Sends the agent's state to the user"""
        state = self.construct_state(agent.state)
        await self._messaging.send_message({"command": "get_agent_state", "data": state})

    def construct_state(self, state: IState) -> Dict[str, Any]:
        """Constructs the agent's state to be sent to the user"""
        result: Dict[str, Any] = {}

        try:

            for key, value in state.items().items():
                if isinstance(value, IAction):
                    value = value.name()
                
                nested_keys = key.split('.')
                branch = result
                for i in range(len(nested_keys) - 1):
                    if nested_keys[i] not in branch:
                        branch[nested_keys[i]] = {}
                    branch = branch[nested_keys[i]]


                branch[nested_keys[-1]] = value
                
        except Exception as e:
            logger.error(
                f"Error constructing state: {e}. This may be due to a non-serializable object in the state or if keyare not nestable by .")

        return result

    async def get_agent_info(self, agent: IAgent, _data: Dict) -> None:
        """Sends the agent's info to the user"""
        await self._messaging.send_message({"command": "get_agent_info", "data": agent.info()})

    async def force_action(self, agent: IAgent, data: Dict) -> None:
        """Forces the agent to perform an action"""
        name = data.get("name")
        args = data.get("args")

        if name is not None:
            agent.state.set_item(
                f"agent.components.{self.info().name}.force_action.name", name)
            agent.state.set_item(
                f"agent.components.{self.info().name}.force_action.args", args)

    async def authenticate(self, agent: IAgent, data: Dict) -> None:
        """Authenticates the user"""

    def info(self) -> EntityInfo:
        return EntityInfo(
            name=self.__class__.__name__,
            version="0.1.0",
            params={
                "address": self._messaging.address()
            })
