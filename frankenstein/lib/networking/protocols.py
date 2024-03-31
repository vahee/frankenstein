from typing import List, Protocol, Dict, Any, runtime_checkable


@runtime_checkable
class IMessaging(Protocol):
    """Interface for messaging services"""

    async def send_message(self, message: Any) -> None:
        """Sends a message"""
        ...

    async def get(self, timeout: float) -> Any:
        """Gets a message, blocking for timeout seconds"""
        ...

    async def get_all(self) -> List[Any]:
        """Gets all messages"""
        ...
    
    async def is_connected(self) -> bool:
        """Returns True if the messaging service is connected, False otherwise"""
        ...

    async def num_messages(self) -> int:
        """Returns the number of messages in the queue"""
        ...

    def address(self) -> Dict[str, Any]:
        """Returns the address of the messaging service"""
        ...
