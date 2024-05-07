class Config:
    # Deployment configurations
    API_VERSION = 'v1'
    API_TITLE = 'spoof.io - Spoofing and Layering Detection API'
    API_DESCRIPTION = 'API for real-time detection of spoofing and layering in algorithmic trading using PPO.'

    # Paths
    RAW_DATA_PATH = 'data/raw/'
    PROCESSED_DATA_PATH = 'data/processed/'
    MISC_DATA_PATH = 'data/misc/'
    MODEL_SAVE_PATH = 'models/'
    OUTPUT_PATH = 'output/'
    EVAL_PATH = 'eval/'

    # Full channel paths
    FULL_CHANNEL_RAW_17HR_PATH = RAW_DATA_PATH + 'FullChannel_GDAX_20220511_17hr.json'
    FULL_CHANNEL_RAW_19HR_PATH = RAW_DATA_PATH + 'FullChannel_GDAX_20220511_19hr.json'
    FULL_CHANNEL_RAW_20HR_PATH = RAW_DATA_PATH + 'FullChannel_GDAX_20220511_20hr.json'
    FULL_CHANNEL_RAW_PATH = RAW_DATA_PATH + 'full_channel.csv'
    FULL_CHANNEL_PROCESSED_PATH = PROCESSED_DATA_PATH + 'full_channel_processed.csv'
    FULL_CHANNEL_ENHANCED_PATH = PROCESSED_DATA_PATH + 'full_channel_enhanced.csv'
    FULL_CHANNEL_SIM_PATH = RAW_DATA_PATH + 'full_channel_sim.csv'

    # Ticker paths
    TICKER_RAW_17HR_PATH = RAW_DATA_PATH + 'Ticker_GDAX_20220511_17hr.json'
    TICKER_RAW_19HR_PATH = RAW_DATA_PATH + 'Ticker_GDAX_20220511_19hr.json'
    TICKER_RAW_20HR_PATH = RAW_DATA_PATH + 'Ticker_GDAX_20220511_20hr.json'
    TICKER_RAW_PATH = RAW_DATA_PATH + 'ticker.csv'
    TICKER_PROCESSED_PATH = PROCESSED_DATA_PATH + 'ticker_processed.csv'
    TICKER_ENHANCED_PATH = PROCESSED_DATA_PATH + 'ticker_enhanced.csv'
    TICKER_SIM_PATH = RAW_DATA_PATH + 'ticker_sim.csv'

    # Model paths
    PPO_POLICY_NETWORK_MODEL_PATH = MODEL_SAVE_PATH + 'ppo_model.pth'

    # Eval paths
    TEST_RESULTS_PATH = EVAL_PATH + 'test_results.html'
    ANOMALY_SCORES_PATH = EVAL_PATH + 'anomaly_scores.png'
    CUMULATIVE_REWARDS_PATH = EVAL_PATH + 'cumulative_rewards.png'
    REWARD_DISTRIBUTION_PATH = EVAL_PATH + 'reward_distribution.png'

    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_MARKET_ENV_PATH = OUTPUT_PATH + 'market_env.log'
    LOG_PPO_POLICY_NETWORK_PATH = OUTPUT_PATH + 'ppo_policy_network.log'
    LOG_TRAIN_PATH = OUTPUT_PATH + 'train.log'
    LOG_TEST_PATH = OUTPUT_PATH + 'test.log'

    # Data preprocessing parameters
    NUMERIC_COLUMNS = ['price', 'size', 'remaining_size', 'remaining_size_change']
    CATEGORICAL_COLUMNS = ['type', 'side', 'reason']
    CATEGORICAL_MAP = {
            'type': ['change', 'done', 'match', 'open', 'received'],
            'side': ['buy', 'sell'],
            'reason': ['canceled', 'filled', 'missing']
        }

    # Feature engineering parameters
    ROLLING_WINDOWS = [5, 10, 15]
    OPERATIONS = ['mean', 'std', 'var']
    HOURS = ['hour_' + str(x) for x in range(15, 20)]

    # Environment simulation parameters
    DEFAULT_SPOOFING_THRESHOLD = 0.8  # Default spoofing threshold for normal runs
    HISTORY_WINDOW_SIZE = 10
    FEATURE_WEIGHTS = {  # Feature weights used in anomaly score calculations
        'order_flow_imbalance': 0.15,
        'cancel_to_received_ratio': 0.15,
        'price_5_std': 0.05,
        'price_10_std': 0.05,
        'price_15_std': 0.05,
        'size_5_var': 0.05,
        'size_10_var': 0.05,
        'size_15_var': 0.05,
        'spread': 0.10,
        'last_size_5_var': 0.05,
        'last_size_10_var': 0.05,
        'hour_of_day': 0.15,  # Higher weight as temporal context is crucial
        'hour_15': 0.025,
        'hour_16': 0.025,
        'hour_17': 0.025,
        'hour_18': 0.025,
        'hour_19': 0.025
    }
    TRAIN_TEST_SPLIT_RATIO = 0.7 # Training configuration
    RANDOM_SEED = 42  # Seed for any random operations to ensure reproducibility

    # Model Checkpointing
    CHECKPOINT_FREQ = 1000  # Frequency of checkpointing model weights
    KEEP_LAST_N_CHECKPOINTS = 3  # Number of checkpoints to keep

    # PPO Model Parameters
    PPO_CONFIG = {
        'learning_rate': 2.5e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5
    }

    # Hyperparameter Tuning
    HYPERPARAMETERS = {
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'batch_size': [32, 64, 128],
        'n_epochs': [10, 20, 30],
        'spoofing_threshold': [0.7, 0.8, 0.9]
    }

    # WEB APP SIMULATION
    BATCH_SIZE = 15  # For feeding and processing live data 
    