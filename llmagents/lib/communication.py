import asyncio as aio
import logging
from typing import List, Dict, Any
from websockets.server import serve
from websockets import ConnectionClosed, WebSocketServerProtocol

from llmagents.lib.protocols import IMessaging

logger = logging.getLogger('Communication')


class WebsocketMessagingServer(IMessaging):
    """A class to represent a websocket connection. 
    This class is used to send and receive messages from the websocket."""

    def __init__(self, host: str, port: int, ping_interval: int, ping_timeout: int):

        self._incoming: List = []
        self._connection: WebSocketServerProtocol | None = None
        self._host: str = host
        self._port: int = port
        aio.ensure_future(serve(self.handler, host, port,
                          ping_interval=ping_interval, ping_timeout=ping_timeout))

    def address(self) -> Dict[str, Any]:
        """Returns the address of the websocket. This is a blocking operation."""
        return {"host": self._host, "port": self._port}

    async def send_message(self, message):
        """Send a message to the websocket. This is a blocking operation."""
        try:
            assert self._connection is not None
            await self._connection.send(message)
        except ConnectionClosed as e:
            logger.error(f"{e}")

    async def get_messages(self):
        """Check the user's messages. This is a blocking operation."""
        incoming: List = self._incoming
        self._incoming = []
        return incoming

    async def num_messages(self):
        """Returns the number of messages in the queue."""
        return len(self._incoming)

    async def is_connected(self):
        """Returns True if the websocket is connected, False otherwise."""
        return self._connection is not None

    async def handler(self, websocket: WebSocketServerProtocol):
        """Handles incoming websocket connections. This is a blocking operation."""
        self._connection = websocket
        logger.info(
            f"Websocket connected: {websocket.remote_address} {'*'* 10} {websocket.local_address}")
        try:
            while True:
                message = await self._connection.recv()
                self._incoming.append(message)
        except ConnectionClosed as e:
            self._incoming = []
            self._connection = None
            logger.error(f"Connection Closed: {e}")
