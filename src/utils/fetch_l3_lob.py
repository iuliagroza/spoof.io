import websocket
import json
import time
import hmac
import hashlib
import base64

def get_auth_headers(api_key, secret_key, passphrase):
    timestamp = str(time.time())
    message = timestamp + 'GET' + '/users/self/verify'
    hmac_key = base64.b64decode(secret_key)
    signature = hmac.new(hmac_key, message.encode(), hashlib.sha256)
    signature_b64 = base64.b64encode(signature.digest()).decode()

    # INPUT CUSTOM RESPONSE INFO
    return {
        'type': 'subscribe',
        'channels': [{'name': 'full', 'product_ids': ['ETH-USD']}],
        'signature': signature_b64,
        'key': api_key,
        'passphrase': passphrase,
        'timestamp': timestamp
    }

def on_open(ws):
    print("Open connection.")

    # INPUT API DATA
    API_KEY = ""
    API_SECRET = ""
    API_PASSPHRASE = ""

    # Subscribe to Level 3 LOB data
    auth_headers = get_auth_headers(API_KEY, API_SECRET, API_PASSPHRASE)
    ws.send(json.dumps(auth_headers))

def on_message(ws, message):
    print("Received message: ")
    print(message)

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, message):
    print("Closed connection.")
    print(message)

def on_ping(ws, message):
    print("Received PING! Sending PONG!")
    ws.send(json.dumps({"type": "pong"}))

def on_pong(ws, message):
    print("Received PONG!")

if __name__ == "__main__":
    # Coinbase Pro WebSocket URL
    socket = "wss://ws-feed.pro.coinbase.com"

    # Create a WebSocket object
    ws = websocket.WebSocketApp(socket,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                on_ping=on_ping,
                                on_pong=on_pong)

    # Run the WebSocket
    ws.run_forever()
