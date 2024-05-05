import csv
import io
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import FullChannel, Ticker
import logging

logger = logging.getLogger(__name__)

class DataConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Accept the WebSocket connection
        await self.accept()

    async def disconnect(self, close_code):
        # Handle disconnecting
        pass

    async def receive(self, text_data=None, bytes_data=None):
        # Handle incoming messages
        if text_data:
            await self.process_csv_data(text_data)

    @database_sync_to_async
    def save_full_channel(self, data):
        try:
            FullChannel.objects.create(**data)
        except Exception as e:
            logger.error(f"Error saving FullChannel data: {e}")

    @database_sync_to_async
    def save_ticker(self, data):
        try:
            Ticker.objects.create(**data)
        except Exception as e:
            logger.error(f"Error saving Ticker data: {e}")

    async def process_csv_data(self, text_data):
        # Convert string to CSV reader
        csv_file = io.StringIO(text_data)
        reader = csv.DictReader(csv_file)

        for row in reader:
            if 'full_channel' in row['type']:
                await self.save_full_channel(row)
            elif 'ticker' in row['type']:
                await self.save_ticker(row)
