from channels.generic.websocket import AsyncWebsocketConsumer
from .simulation import MarketEnvironment
import json
import asyncio

class TradingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.market_environment = MarketEnvironment()

        # Run the market simulation and send updates to the client
        await self.market_environment.run(self.send_update)

    async def disconnect(self, close_code):
        # Handle disconnection
        pass

    async def receive(self, text_data):
        # Process messages received from the websocket
        text_data_json = json.loads(text_data)
        action = text_data_json.get('action')

        # Simulate an action taken by the user
        if action is not None:
            await self.market_environment.step(int(action))

    async def send_update(self, message):
        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'state': message.get('state', []),
            'reward': message.get('reward'),
            'done': message.get('done'),
            'anomaly_score': message.get('anomaly_score'),
            'spoofing_threshold': message.get('spoofing_threshold'),
            'action_taken': message.get('action_taken')
        }))
