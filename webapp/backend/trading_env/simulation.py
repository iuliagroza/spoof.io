import torch
from torch.distributions import Categorical
import asyncio
from datetime import datetime
import json
from channels.layers import get_channel_layer
import pandas as pd
from trading_env.config import Config
from trading_env.preprocess_data import preprocess_full_channel_data, preprocess_ticker_data
from trading_env.extract_features import extract_full_channel_features, extract_ticker_features
from trading_env.market_env import MarketEnvironment
from trading_env.ppo_policy_network import PPOPolicyNetwork
from trading_env.utils.save_data import save_data
from trading_env.utils.log_config import setup_logger


logger = setup_logger('test', Config.LOG_TEST_PATH)


async def send_order(order, is_spoof=False):
    channel_layer = get_channel_layer()

    if not isinstance(order, dict):
        if hasattr(order, 'to_dict'):
            order_dict = order.to_dict()
        else:
            raise TypeError("Order data must be a dictionary or convertible to a dictionary.")
    else:
        order_dict = order

    order_dict = {k: (None if pd.isna(v) else v) for k, v in order_dict.items()}

    for key, value in order_dict.items():
        if isinstance(value, (pd.Timestamp, datetime)):
            order_dict[key] = value.isoformat()

    if is_spoof:
        order_dict.update({
            'is_spoofing': True,
            'anomaly_score': order['anomaly_score'],
            'spoofing_threshold': order['spoofing_threshold']
        })

    await channel_layer.group_send(
        'order_group',
        {
            'type': 'order.message',
            'message': json.dumps(order_dict)
        }
    )


def load_model(model_path, num_features, num_actions):
    """
    Load the trained PPO policy network model from a given file path.

    Args:
        model_path (str): The file path where the model is saved.
        num_features (int): Number of input features for the model.
        num_actions (int): Number of actions the model can take.

    Returns:
        PPOPolicyNetwork: The loaded and trained policy network ready for inference.
    """
    model = PPOPolicyNetwork(num_features, num_actions)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


async def test_model(env, model):
    """
    Test the model by simulating its interaction with the environment and log results.

    Args:
        env (MarketEnvironment): An instance of the market environment to test the model on.
        model (PPOPolicyNetwork): The trained PPO model to be tested.

    This function runs a simulation in the environment using the trained model. It logs
    each step's action, reward, anomaly score, and spoofing threshold. It also logs the
    total reward accumulated over all steps at the end of the test.
    """
    try:
        states = []
        actions = []
        rewards = []
        anomaly_scores = []
        spoofing_thresholds = []

        state = env.reset()
        total_reward = 0
        steps = 0

        while not env.done:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(state)
                dist = Categorical(logits=logits)
                action = dist.sample()

            state, transaction_data, reward, done, anomaly_score, spoofing_threshold = env.step(action.item())
            logger.info(transaction_data)
            if transaction_data is not None and not pd.isna(transaction_data['order_id'])  and action.item() == 1:
                transaction_data.update({'anomaly_score': anomaly_score, 'spoofing_threshold': spoofing_threshold})
                await send_order(transaction_data, is_spoof=True)

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            anomaly_scores.append(anomaly_score)
            spoofing_thresholds.append(spoofing_threshold)
            total_reward += reward
            steps += 1
            logger.info(f"Step: {steps}, Action: {action.item()}, Reward: {reward}, Anomaly Score: {anomaly_score}, Spoofing Threshold: {spoofing_threshold}")

        logger.info(f"Test completed. Total Reward: {total_reward}, Total Steps: {steps}")
    except Exception as e:
        logger.error(f"An error occurred during the test: {e}")


async def simulate_market_data():
    # Load the data
    full_channel_data = pd.read_csv(Config.FULL_CHANNEL_SIM_PATH)
    ticker_data = pd.read_csv(Config.TICKER_SIM_PATH)

    # Convert 'time' to datetime and sort by it if necessary
    full_channel_data['time'] = pd.to_datetime(full_channel_data['time'])
    ticker_data['time'] = pd.to_datetime(ticker_data['time'])

    # Calculate the delays
    full_channel_data['delay'] = full_channel_data['time'].diff().dt.total_seconds().fillna(0)
    ticker_data['delay'] = ticker_data['time'].diff().dt.total_seconds().fillna(0)

    # Initialize batches
    full_channel_batch = []
    ticker_batch = []

    # Start processing each row as a 'real-time' feed
    full_channel_iter = full_channel_data.iterrows()
    ticker_iter = ticker_data.iterrows()

    try:
        while True:
            # Get next entry from each data source
            _, full_channel_row = next(full_channel_iter)
            _, ticker_row = next(ticker_iter)

            # Sleep to simulate real-time data feed
            await asyncio.sleep(full_channel_row['delay'])
            await asyncio.sleep(ticker_row['delay'])

            # Append to batch without the 'delay' column
            full_channel_row_without_delay = full_channel_row.drop('delay')
            ticker_row_without_delay = ticker_row.drop('delay')
            full_channel_batch.append(full_channel_row_without_delay)
            ticker_batch.append(ticker_row_without_delay)

            # Send simulated order to frontend order box
            await send_order(full_channel_row.drop('delay'))

            # If batch is ready, process it
            if len(full_channel_batch) == Config.BATCH_SIZE:
                full_channel_df = pd.DataFrame(full_channel_batch)
                ticker_df = pd.DataFrame(ticker_batch)
                save_data(full_channel_df, ticker_df, Config.MISC_DATA_PATH + 'full_channel_df.csv', Config.MISC_DATA_PATH + 'ticker_df.csv')

                # Preprocess and feature engineer
                processed_full_channel = preprocess_full_channel_data(full_channel_df)
                processed_ticker = preprocess_ticker_data(ticker_df)
                save_data(processed_full_channel, processed_ticker, Config.MISC_DATA_PATH + 'full_channel_prep.csv', Config.MISC_DATA_PATH + 'ticker_prep.csv')
                enhanced_full_channel = extract_full_channel_features(processed_full_channel)
                enhanced_ticker = extract_ticker_features(processed_ticker)

                save_data(enhanced_full_channel, enhanced_ticker, Config.MISC_DATA_PATH + 'full_channel_output.csv', Config.MISC_DATA_PATH + 'ticker_output.csv')

                env = MarketEnvironment(initial_index=0, full_channel_data=enhanced_full_channel, ticker_data=enhanced_ticker, train=False)
                model = load_model(Config.PPO_POLICY_NETWORK_MODEL_PATH, len(env.reset()), 2)
                await test_model(env, model)

                # Reset batches
                full_channel_batch = []
                ticker_batch = []

    except StopIteration:
        print("End of data.")

    except Exception as e:
        print(f"An error occurred: {e}")


def start_simulation():
    simulate_market_data()

if __name__ == "__main__":
    start_simulation()
