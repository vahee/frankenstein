import asyncio as aio
import logging
from typing import Dict, Any, List
from websockets.server import serve
from websockets import ConnectionClosed, WebSocketServerProtocol
from orjson import loads, dumps, JSONDecodeError, JSONEncodeError

from frankenstein.lib.networking.protocols import IMessaging

logger = logging.getLogger('Communication')


def default(obj: Any) -> Any:
    if hasattr(obj,'__dict__'):
        return obj.__dict__()
    else:
        return obj


class WebsocketMessagingJsonServer(IMessaging):
    """A class to represent a websocket connection. 
    This class is used to send and receive messages from the websocket."""

    def __init__(self, host: str, port: int, ping_interval: int, ping_timeout: int):

        self._incoming: aio.Queue = aio.Queue()
        self._connection: WebSocketServerProtocol | None = None
        self._host: str = host
        self._port: int = port
        aio.ensure_future(serve(self.handler, host, port,
                          ping_interval=ping_interval, ping_timeout=ping_timeout))

    def address(self) -> Dict[str, Any]:
        """Returns the address of the websocket. This is a blocking operation."""
        return {"host": self._host, "port": self._port}

    async def send_message(self, message: Any):
        """Send a message to the websocket. This is a blocking operation."""
        try:
            assert self._connection is not None
            try:
                message = dumps(message, default=default).decode()
            except JSONEncodeError as e:
                logger.error(f"Invalid message: {e}")
                return
            await self._connection.send(message)
        except ConnectionClosed as e:
            logger.error(f"{e}")

    async def get_all(self) -> List[Any]:
        """Returns all the messages in the queue. This is a blocking operation."""
        messages = []
        for _ in range(self._incoming.qsize()):
            messages.append(await self._incoming.get())
        return messages
    
    async def get(self, timeout: float = 0) -> Any:
        """Gets a message from the websocket. This is a blocking operation."""
        try:
            return await aio.wait_for(self._incoming.get(), timeout=timeout)
        except aio.TimeoutError:
            return None

    async def num_messages(self):
        """Returns the number of messages in the queue."""
        return self._incoming.qsize()

    async def is_connected(self):
        """Returns True if the websocket is connected, False otherwise."""
        return self._connection is not None

    async def handler(self, websocket: WebSocketServerProtocol):
        """Handles incoming websocket connections. This is a blocking operation."""
        if self._connection is not None:
            await websocket.close(reason="Another connection is already open")
            return
        self._connection = websocket
        logger.info(
            f"Websocket connected: {websocket.remote_address} {'*'* 10} {websocket.local_address}")
        try:
            while True:
                message = await self._connection.recv()
                try:
                    message = loads(message)
                except JSONDecodeError as e:
                    logger.error(f"Invalid message: {e}")
                await self._incoming.put(message)
        except ConnectionClosed as e:
            self._connection = None
            logger.error(f"Connection Closed: {e}")
