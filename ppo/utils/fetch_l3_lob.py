import websocket
import json
import time
import hmac
import hashlib
import base64

# Configuration variables
API_KEY = ''
API_SECRET = ''
API_PASSPHRASE = ''
PRODUCT_IDS = ['ETH-USD']
CHANNELS = [{'name': 'full', 'product_ids': PRODUCT_IDS}]
SOCKET_URL = 'wss://ws-feed.pro.coinbase.com' # Coinbase Pro WebSocket URL


def get_auth_headers(api_key, secret_key, passphrase):
    """
    Generate authentication headers for Coinbase Pro WebSocket API.

    Args:
        api_key (str): Your Coinbase Pro API key.
        secret_key (str): Your Coinbase Pro API secret key.
        passphrase (str): Your Coinbase Pro API passphrase.

    Returns:
        dict: The authentication headers containing the necessary information.
    """
    timestamp = str(time.time())
    message = timestamp + 'GET' + '/users/self/verify'
    hmac_key = base64.b64decode(secret_key)
    signature = hmac.new(hmac_key, message.encode(), hashlib.sha256)
    signature_b64 = base64.b64encode(signature.digest()).decode()

    return {
        'type': 'subscribe',
        'channels': CHANNELS,
        'signature': signature_b64,
        'key': api_key,
        'passphrase': passphrase,
        'timestamp': timestamp
    }


def on_open(ws):
    """
    Callback function when the WebSocket connection is opened.

    Args:
        ws (WebSocketApp): The WebSocketApp object.
    """
    print("Connection opened.")

    # Subscribe to the configured channels
    auth_headers = get_auth_headers(API_KEY, API_SECRET, API_PASSPHRASE)
    ws.send(json.dumps(auth_headers))


def on_message(ws, message):
    """
    Callback function when a message is received from the WebSocket.

    Args:
        ws (WebSocketApp): The WebSocketApp object.
        message (str): The received message.
    """
    print("Received message:")
    print(message)


def on_error(ws, error):
    """
    Callback function when an error occurs with the WebSocket.

    Args:
        ws (WebSocketApp): The WebSocketApp object.
        error (str): The error message.
    """
    print(f"Error: {error}")


def on_close(ws, close_status_code, close_msg):
    """
    Callback function when the WebSocket connection is closed.

    Args:
        ws (WebSocketApp): The WebSocketApp object.
        close_status_code (int): The status code of the close event.
        close_msg (str): The close message.
    """
    print("Connection closed.")
    print(f"Status code: {close_status_code}")
    print(f"Close message: {close_msg}")


def on_ping(ws, message):
    """
    Callback function when a ping message is received from the WebSocket.

    Args:
        ws (WebSocketApp): The WebSocketApp object.
        message (str): The ping message.
    """
    print("Received PING! Sending PONG...")
    ws.send(json.dumps({'type': 'pong'}))


def on_pong(ws, message):
    """
    Callback function when a pong message is received from the WebSocket.

    Args:
        ws (WebSocketApp): The WebSocketApp object.
        message (str): The pong message.
    """
    print("Received PONG!")


if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        SOCKET_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_ping=on_ping,
        on_pong=on_pong
    )

    ws.run_forever()
