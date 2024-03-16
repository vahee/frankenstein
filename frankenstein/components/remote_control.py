from typing import Dict, Any, Optional
import orjson
import logging

from agentopy import WithStateMixin, WithActionSpaceMixin, IAgentComponent, IAgent, IAction, IState, EntityInfo

from frankenstein.lib.networking.communication import IMessaging

logger = logging.getLogger('[Component][RemoteControl]')


class RemoteControl(WithStateMixin, WithActionSpaceMixin, IAgentComponent):
    """Implements a remote control component that allows to control the agent remotely."""

    def __init__(self, messaging: IMessaging):
        super().__init__()
        self._messaging: IMessaging = messaging
        self._agent_state_subscription: Dict[str, Any] | None = None

    async def on_heartbeat(self, agent: IAgent) -> None:

        if not await self._messaging.is_connected():
            self._agent_state_subscription = None
            return

        if self._agent_state_subscription is not None:
            await self.get_agent_state(agent, self._agent_state_subscription.get("message"))

        messages = await self._messaging.get_messages()

        for message in messages:
            await self.process_message(agent, message)

    async def process_message(self, agent: IAgent, raw_message: str) -> None:
        """Processes the message"""
        try:
            message = orjson.loads(raw_message)
        except orjson.JSONDecodeError:
            await self._messaging.send_message(orjson.dumps(
                {"error": "Invalid message"}).decode())
            return

        await {
            "get_agent_state": self.get_agent_state,
            "subscribe_agent_state": self.subscribe_agent_state,
            "unsubscribe_agent_state": self.unsubscribe_agent_state,
            "get_agent_info": self.get_agent_info,
            "force_action": self.force_action,
            "authenticate": self.authenticate,
        }.get(message.get("command"), self.do_nothing)(agent, message)

    async def do_nothing(self, agent: IAgent, message: Dict) -> None:
        """Does nothing"""

    async def subscribe_agent_state(self, _agent: IAgent, message: Optional[Dict[str, Any]]) -> None:
        """Subscribes to the agent's state"""
        self._agent_state_subscription = {
            "message": message
        }

    async def unsubscribe_agent_state(self, _agent: IAgent, _message: Optional[Dict[str, Any]]) -> None:
        """Unsubscribes from the agent's state"""
        self._agent_state_subscription = None

    async def get_agent_state(self, agent: IAgent, _: Optional[Dict[str, Any]]) -> None:
        """Sends the agent's state to the user"""
        state = self.construct_state(agent.state)

        await self._messaging.send_message(orjson.dumps({"command": "get_agent_state", "data": state}).decode())

    def construct_state(self, state: IState) -> Dict[str, Any]:
        """Constructs the agent's state to be sent to the user"""
        result: Dict[str, Any] = {}

        try:

            for key, value in state.items().items():
                if isinstance(value, IAction):
                    value = value.name()

                nested_keys = key.split('/')
                branch = result
                for i in range(len(nested_keys) - 1):
                    if nested_keys[i] not in branch:
                        branch[nested_keys[i]] = {}
                    branch = branch[nested_keys[i]]

                branch[nested_keys[-1]] = value
        except Exception as e:
            logger.error(
                f"Error constructing state: {e}. This may be due to a non-serializable object in the state or if keyare not nestable by /")

        return result

    async def get_agent_info(self, agent: IAgent, _message: Dict) -> None:
        """Sends the agent's info to the user"""
        await self._messaging.send_message(orjson.dumps({"command": "get_agent_info", "data": agent.info()}).decode())

    async def force_action(self, agent: IAgent, message: Dict) -> None:
        """Forces the agent to perform an action"""
        action = message.get("action")
        args = message.get("args")

        if action is not None:
            agent.state.set_item(
                "agent/components/remote_control/force_action/name", action)
            agent.state.set_item(
                "agent/components/remote_control/force_action/args", args)

    async def authenticate(self, agent: IAgent, message: Dict) -> None:
        """Authenticates the user"""

    def info(self) -> EntityInfo:
        return EntityInfo(
            name="RemoteControl",
            version="0.1.0",
            params={
                "address": self._messaging.address()
            })
