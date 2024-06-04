import json
import os
import pytest
import pandas as pd
from datetime import datetime
from trading_env.market_env import MarketEnvironment
from trading_env.ppo_policy_network import PPOPolicyNetwork
from trading_env.config import Config
from trading_env.simulation import load_model, test_model, simulate_market_data, send_order
from asgiref.testing import ApplicationCommunicator
from channels.layers import get_channel_layer
from channels.testing import WebsocketCommunicator

@pytest.mark.asyncio
async def test_send_order():
    channel_layer = get_channel_layer()
    communicator = ApplicationCommunicator(channel_layer, "order_group")

    order = {
        'order_id': '12345',
        'time': datetime.now(),
        'anomaly_score': 0.7,
        'spoofing_threshold': 0.6
    }

    await send_order(order, is_spoof=True)

    message = await communicator.receive_json_from()

    assert message['type'] == 'order.message'
    assert message['message'] == json.dumps(order)

@pytest.mark.asyncio
async def test_load_model():
    num_features = 10
    num_actions = 2
    model_path = Config.PPO_POLICY_NETWORK_MODEL_PATH

    model = load_model(model_path, num_features, num_actions)

    assert isinstance(model, PPOPolicyNetwork)
    assert model.fc1.in_features == num_features
    assert model.fc2.out_features == num_actions

@pytest.mark.asyncio
async def test_test_model():
    data = pd.DataFrame({
        'time': [datetime.now()] * 100,
        'price': [100.0] * 100,
        'size': [1.0] * 100
    })
    env = MarketEnvironment(initial_index=0, full_channel_data=data, ticker_data=data, train=False)
    model = PPOPolicyNetwork(env.observation_space.shape[0], env.action_space.n)

    await test_model(env, model)

    assert env.total_reward >= 0 
    assert len(env.actions) > 0  

@pytest.mark.asyncio
async def test_simulate_market_data():
    await simulate_market_data()

    assert os.path.exists(Config.MISC_DATA_PATH + 'full_channel_output.csv')
    assert os.path.exists(Config.MISC_DATA_PATH + 'ticker_output.csv')
