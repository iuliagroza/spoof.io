import torch
from torch.distributions import Categorical
from src.market_env import MarketEnvironment
from src.ppo_policy_network import PPOPolicyNetwork
from src.utils.log_config import setup_logger
from src.config import Config


# Set up logging to save environment logs for debugging purposes
logger = setup_logger('test', Config.LOG_TEST_PATH)


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


def test_model(env, model):
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
        state = env.reset()
        total_reward = 0
        steps = 0

        while not env.done:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(state)
                dist = Categorical(logits=logits)
                action = dist.sample()

            state, reward, done, anomaly_score, spoofing_threshold = env.step(action.item())
            total_reward += reward
            steps += 1
            logger.info(f"Step: {steps}, Action: {action.item()}, Reward: {reward}, Anomaly Score: {anomaly_score}, Spoofing Threshold: {spoofing_threshold}")

        logger.info(f"Test completed. Total Reward: {total_reward}, Total Steps: {steps}")
    except Exception as e:
        logger.error(f"An error occurred during the test: {e}")


if __name__ == "__main__":
    try:
        env = MarketEnvironment(train=False)
        model = load_model(Config.PPO_POLICY_NETWORK_MODEL_PATH, len(env.reset()), 2)
        test_model(env, model)
    except Exception as e:
        logger.error(f"Failed to start the testing process: {e}")
