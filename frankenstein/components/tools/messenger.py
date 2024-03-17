from typing import Any, List
import orjson

from agentopy import WithStateMixin, WithActionSpaceMixin, Action, ActionResult, IEnvironmentComponent, EntityInfo

from frankenstein.lib.networking.communication import IMessaging


class Messenger(WithStateMixin, WithActionSpaceMixin, IEnvironmentComponent):
    """Implements a communication component that uses websockets to send and receive messages."""

    def __init__(self, messaging: IMessaging):
        super().__init__()
        self.action_space.register_actions([
            Action(
                "send_message", "use this action to send a text message to the user, use it as the main way to interact with the user (input param: 'message' of string type)", self.send_message),
            Action(
                "fetch_messages", "use this action to check messages user sent to you, prioritise acting on those", self.get_messages),
        ])

        self._messaging: IMessaging = messaging

    async def send_message(self, message: Any) -> ActionResult:
        """Sends a message to the user. This is a blocking operation."""
        if not await self._messaging.is_connected():
            return ActionResult(value="No connection", success=False)

        await self._messaging.send_message(orjson.dumps(message).decode())

        return ActionResult(value="OK", success=True)

    async def get_messages(self) -> ActionResult:
        """Gets the user's messages. This is a blocking operation."""
        if await self._messaging.is_connected():
            messages: List = await self._messaging.get_messages()
            return ActionResult(value={f"Message {i}": msg for i, msg in enumerate(messages)}, success=True)
        else:
            return ActionResult(value="No messages", success=True)

    async def on_tick(self) -> None:
        num_messages: int = await self._messaging.num_messages()
        self._state.set_item(
            "status", {"num_new_messages": num_messages})

    def info(self) -> EntityInfo:
        return EntityInfo(
            name="Messenger",
            version="0.1.0",
            params={
                "address": self._messaging.address()
            }
        )
