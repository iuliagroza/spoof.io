import os
import django
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.core.asgi import get_asgi_application
from django.urls import path
from trading_env.consumers import OrderConsumer

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

application = ProtocolTypeRouter({
    "http": get_asgi_application(),  # Django's ASGI application to handle traditional HTTP requests
    "websocket": AuthMiddlewareStack(  # AuthMiddlewareStack wraps the ASGI application to manage the WebSocket connection with authentication
        URLRouter([
            path("ws/orders/", OrderConsumer.as_asgi()),  # Route for WebSocket connections to be handled by OrderConsumer
        ])
    ),
})
