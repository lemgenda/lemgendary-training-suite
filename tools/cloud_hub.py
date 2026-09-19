"""LemGendary Cloud Link: WebSockets Coordinator Hub.

Tracks epoch syncs, learning rate recoils, and federated coordination events.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class LemGendaryCloudHub:
    """Lightweight WebSockets Coordinator Hub for distributed training nodes."""

    def __init__(self) -> None:
        self.connected_nodes: set[Any] = set()
        self.global_epoch: int = 0
        self.global_lr: float = 0.0
        self.node_states: dict[Any, Any] = {}
        self.accumulated_gradients: int = 0

    async def register(self, websocket: Any) -> None:
        """Register newly connected node."""
        self.connected_nodes.add(websocket)
        logging.info("Node connected. Total nodes: %d", len(self.connected_nodes))
        await websocket.send(json.dumps({
            "type": "SYNC_STATE",
            "global_epoch": self.global_epoch,
            "global_lr": self.global_lr,
        }))

    async def unregister(self, websocket: Any) -> None:
        """Unregister disconnected node."""
        self.connected_nodes.remove(websocket)
        if websocket in self.node_states:
            del self.node_states[websocket]
        logging.info("Node disconnected. Total nodes: %d", len(self.connected_nodes))

    async def handle_message(self, websocket: Any, message: str) -> None:
        """Process incoming coordinator control messages."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "NODE_HEARTBEAT":
                self.node_states[websocket] = data.get("params", {})
                logging.info("Heartbeat from node. Parameters: %s", data.get("params", {}))

            elif msg_type == "EPOCH_SYNC":
                logging.info("Epoch sync received from node: %s", data.get("epoch"))
                self.global_epoch = max(self.global_epoch, data.get("epoch", 0))

            elif msg_type == "LR_RECOIL":
                logging.warning("Learning Rate Recoil broadcasted by node: %s", data.get("lr"))
                self.global_lr = data.get("lr")
                websockets.broadcast(self.connected_nodes, json.dumps({
                    "type": "LR_RECOIL_SYNC",
                    "global_lr": self.global_lr,
                }))

            elif msg_type == "GRADIENT_PUSH":
                self.accumulated_gradients += 1
                logging.info("Federated gradient chunk received. Total: %d", self.accumulated_gradients)
                if self.accumulated_gradients >= max(1, len(self.connected_nodes)):
                    logging.info("Broadcasting unified average-sync gradient vector to all nodes.")
                    websockets.broadcast(self.connected_nodes, json.dumps({
                        "type": "GRADIENT_AVERAGE_SYNC",
                        "status": "success",
                    }))
                    self.accumulated_gradients = 0

        except json.JSONDecodeError:
            logging.error("Failed to decode message.")

    async def handler(self, websocket: Any) -> None:
        """Connection handler loop."""
        await self.register(websocket)
        try:
            async for message in websocket:
                await self.handle_message(websocket, str(message))
        finally:
            await self.unregister(websocket)


async def main() -> None:
    """Start websocket coordinator server."""
    hub = LemGendaryCloudHub()
    async with websockets.serve(hub.handler, "0.0.0.0", 8765):
        logging.info("LemGendary Cloud Link Coordinator Hub started on ws://0.0.0.0:8765")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
