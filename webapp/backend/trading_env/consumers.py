from channels.generic.websocket import AsyncWebsocketConsumer
import asyncio
from trading_env.simulation import simulate_market_data

class OrderConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_group_name = 'order_group'

        # Join the room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

        # Start the simulation as soon as the WebSocket connection is accepted
        asyncio.create_task(self.start_simulation())

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def order_message(self, event):
        message = event['message']
        await self.send(text_data=message)

    # Function to handle the start and management of the simulation
    async def start_simulation(self):
        await simulate_market_data()
