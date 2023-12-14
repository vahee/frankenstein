from typing import List, Protocol, Dict, Any, runtime_checkable


@runtime_checkable
class IMessaging(Protocol):
    """Interface for messaging services"""

    async def send_message(self, message: str) -> None:
        """Sends a message"""
        ...

    async def get_messages(self) -> List[str]:
        """Checks the user's messages"""
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
