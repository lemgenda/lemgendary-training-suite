import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class LemGendaryCloudHub:
    """
    LemGendary Cloud Link: Lightweight WebSockets Coordinator Hub
    Tracks epoch syncs, learning rate recoils, and handles federated gradient accumulations.
    """
    def __init__(self):
        self.connected_nodes = set()
        self.global_epoch = 0
        self.global_lr = 0.0
        self.node_states = {}
        # We simulate federated gradient sync states
        self.accumulated_gradients = 0

    async def register(self, websocket):
        self.connected_nodes.add(websocket)
        logging.info(f"Node connected. Total nodes: {len(self.connected_nodes)}")
        # Send current global state
        await websocket.send(json.dumps({
            "type": "SYNC_STATE",
            "global_epoch": self.global_epoch,
            "global_lr": self.global_lr
        }))

    async def unregister(self, websocket):
        self.connected_nodes.remove(websocket)
        if websocket in self.node_states:
            del self.node_states[websocket]
        logging.info(f"Node disconnected. Total nodes: {len(self.connected_nodes)}")

    async def handle_message(self, websocket, message):
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "NODE_HEARTBEAT":
                self.node_states[websocket] = data.get("params", {})
                logging.info(f"Heartbeat from node. Parameters: {data.get('params', {})}")
                
            elif msg_type == "EPOCH_SYNC":
                # A node finished an epoch and wants to sync
                logging.info(f"Epoch sync received from node: {data.get('epoch')}")
                self.global_epoch = max(self.global_epoch, data.get("epoch", 0))
                
            elif msg_type == "LR_RECOIL":
                logging.warning(f"Learning Rate Recoil broadcasted by a node! New LR: {data.get('lr')}")
                self.global_lr = data.get("lr")
                # Broadcast recoil to all nodes
                websockets.broadcast(self.connected_nodes, json.dumps({
                    "type": "LR_RECOIL_SYNC",
                    "global_lr": self.global_lr
                }))
                
            elif msg_type == "GRADIENT_PUSH":
                # Simulated federated gradient accumulation bypass
                self.accumulated_gradients += 1
                logging.info(f"Federated Gradient chunk received. Total chunks: {self.accumulated_gradients}")
                # Broadcast unified sync when we hit node quorum (or just 1 for simulation)
                if self.accumulated_gradients >= max(1, len(self.connected_nodes)):
                    logging.info("Broadcasting unified average-sync gradient vector to all nodes.")
                    websockets.broadcast(self.connected_nodes, json.dumps({
                        "type": "GRADIENT_AVERAGE_SYNC",
                        "status": "success"
                    }))
                    self.accumulated_gradients = 0

        except json.JSONDecodeError:
            logging.error("Failed to decode message.")

    async def handler(self, websocket):
        await self.register(websocket)
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        finally:
            await self.unregister(websocket)

async def main():
    hub = LemGendaryCloudHub()
    # Bind to 0.0.0.0 to allow WAN/Edge nodes
    async with websockets.serve(hub.handler, "0.0.0.0", 8765):
        logging.info(" [CLOUD] LemGendary Cloud Link Coordinator Hub started on ws://0.0.0.0:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
