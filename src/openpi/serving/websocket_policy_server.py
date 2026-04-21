import asyncio
import logging
import traceback
from pprint import pformat
from typing import Any

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.frames


def _normalize_prompt(prompt: Any) -> Any:
    """
    Normalize prompt-like inputs to a string when possible.

    Supported cases:
    - None -> None
    - list[str] with one element -> that element
    - list/tuple with multiple elements -> join by spaces
    - scalar-like objects with .item() -> .item()
    - otherwise return as-is
    """
    if prompt is None:
        return None

    if isinstance(prompt, (list, tuple)):
        if len(prompt) == 0:
            return ""
        if len(prompt) == 1:
            return prompt[0]
        return " ".join(map(str, prompt))

    if hasattr(prompt, "item"):
        try:
            return prompt.item()
        except Exception:
            pass

    return prompt


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))
        logging.info("Sent metadata to client: %s", pformat(self._metadata))

        while True:
            try:
                raw = await websocket.recv()

                logging.info(
                    "Received raw websocket frame from %s, type=%s, size=%s bytes",
                    websocket.remote_address,
                    type(raw).__name__,
                    len(raw) if hasattr(raw, "__len__") else "unknown",
                )

                obs = msgpack_numpy.unpackb(raw)

                if isinstance(obs, dict) and "prompt" in obs:
                    before = obs["prompt"]
                    after = _normalize_prompt(before)
                    if after is not before:
                        logging.warning(
                            "Prompt normalized from %s to %s",
                            type(before).__name__,
                            type(after).__name__,
                        )
                    obs["prompt"] = after

                logging.info(
                    "Decoded obs keys: %s",
                    list(obs.keys()) if isinstance(obs, dict) else type(obs),
                )
                logging.info("Decoded obs preview:\n%s", pformat(obs, width=120, compact=True))

                action = self._policy.infer(obs)

                logging.info("Policy action preview:\n%s", pformat(action, width=120, compact=True))

                await websocket.send(packer.pack(action))

            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logging.exception("Error while handling websocket request")
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise