from typing import Any
import time
import logging
from agentopy import WithActionSpaceMixin, Action, ActionResult, IEnvironmentComponent, EntityInfo, State, IState
import asyncio as aio
logger = logging.getLogger(__name__)

class Messenger(WithActionSpaceMixin, IEnvironmentComponent):
    """Implements a communication component that uses websockets to send and receive messages."""

    def __init__(self):
        super().__init__()
        
        self._messages = {}
        
        self.action_space.register_actions([
            Action(
                "send_message", "use this action to send a text message to the user, use it as the main way to interact with the user (input param: 'message' of string type)", self.send_message, self.info()),
            Action(
                "fetch_new_messages", "use this action to check messages user sent to you, prioritise acting on those", self.get_new_messages, self.info()),
            Action("get_message_history", "use this action to get the history of messages sent and received", self.get_message_history, self.info())
        ])
            
    async def send_message(self, message: Any, caller_context: IState) -> ActionResult:
        """Sends a message to the thread."""
        assert caller_context is not None, "Caller context is required"
        login = caller_context.get_item("login")
        msg = {
            "type": "text",
            "data": {
                "from": login,
                "ts": time.time(),
                "data": message
            }
        }
        
        for user_login, messages in self._messages.items():
            m = msg.copy()
            m["read"] = user_login == login
            messages.append(m)
        
        return ActionResult(value="OK", success=True)

    async def get_new_messages(self, *, caller_context: IState) -> ActionResult:
        """Gets unread messages. This is a blocking operation."""
        assert caller_context is not None, "Caller context is required"
        login = caller_context.get_item("login")
        
        new_messages = [ msg for msg in self._messages.get(login, []) if not msg.get("read") ]
        
        # mark messages as read
        for msg in new_messages:
            msg["read"] = True
        
        return ActionResult(value=new_messages, success=True)
        
    async def get_message_history(self, *, caller_context: IState) -> ActionResult:
        """Gets the message history. This is a blocking operation."""
        
        login = caller_context.get_item("login")
        
        return ActionResult(value=self._messages.get(login, []), success=True)

    async def tick(self) -> None:
        await aio.sleep(10)
            
    async def observe(self, caller_context: IState) -> State:
        
        assert caller_context is not None, "Observer context is required"
        
        login = caller_context.get_item("login")
        
        if login and login not in self._messages:
            self._messages[login] = []
        
        # count new messages
        num_new_messages = len([ msg for msg in self._messages.get(login, []) if not msg.get("read") ])
        
        for msg in self._messages[login]:
            msg["read"] = True
        
        state = State()
        data = {
            "num_new_messages": num_new_messages,
            "messages__": self._messages.get(login, [])
        }
        state.set_item("status", data)
        
        return state
        
    def info(self) -> EntityInfo:
        return EntityInfo(
            name=self.__class__.__name__,
            version="0.1.0",
            params={}
        )
