from typing import Any, Dict
import time
import logging
from agentopy import WithActionSpaceMixin, Action, ActionResult, IEnvironmentComponent, EntityInfo, State, IState

logger = logging.getLogger(__name__)

class Messenger(WithActionSpaceMixin, IEnvironmentComponent):
    """Implements a communication component that uses websockets to send and receive messages."""

    def __init__(self):
        super().__init__()
        
        self._num_read_messages: Dict[str, int] = {}
        self._messages = []
        
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
        
        self._messages.append(msg["data"])
        
        return ActionResult(value="OK", success=True)

    async def get_new_messages(self, *, caller_context: IState) -> ActionResult:
        """Gets unread messages. This is a blocking operation."""
        assert caller_context is not None, "Caller context is required"
        login = caller_context.get_item("login")
        num_read_messages = self._num_read_messages.get(login, 0)
        new_messages = self._messages[num_read_messages:]
        self._num_read_messages[login] = len(self._messages)
        return ActionResult(value={f"Message {i}": msg.get("data") for i, msg in enumerate(new_messages)}, success=True)
        
    async def get_message_history(self, *, caller_context: IState) -> ActionResult:
        """Gets the message history. This is a blocking operation."""
        return ActionResult(value=self._messages, success=True)

    async def tick(self) -> None:
        ...
            
    async def observe(self, caller_context: IState) -> State:
        
        assert caller_context is not None, "Observer context is required"
        
        login = caller_context.get_item("login")
        
        num_new_messages = len(self._messages) - self._num_read_messages.get(login, 0)
        
        state = State()
        state.set_item("status", {"num_new_messages": num_new_messages})
        
        return state
        
    def info(self) -> EntityInfo:
        return EntityInfo(
            name=self.__class__.__name__,
            version="0.1.0",
            params={}
        )
