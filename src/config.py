class Config:
    # Deployment configurations
    API_VERSION = 'v1'
    API_TITLE = 'spoof.io - Spoofing and Layering Detection API'
    API_DESCRIPTION = 'API for real-time detection of spoofing and layering in algorithmic trading using PPO.'

    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

    # Paths
    RAW_DATA_PATH = 'data/raw/'
    PROCESSED_DATA_PATH = 'data/processed/'
    MISC_DATA_PATH = 'data/misc/'
    MODEL_SAVE_PATH = 'models/'

    # Full channel paths
    FULL_CHANNEL_RAW_17HR_PATH = RAW_DATA_PATH + 'FullChannel_GDAX_20220511_17hr.json'
    FULL_CHANNEL_RAW_19HR_PATH = RAW_DATA_PATH + 'FullChannel_GDAX_20220511_19hr.json'
    FULL_CHANNEL_RAW_20HR_PATH = RAW_DATA_PATH + 'FullChannel_GDAX_20220511_20hr.json'
    FULL_CHANNEL_RAW_PATH = RAW_DATA_PATH + 'full_channel.csv'
    FULL_CHANNEL_PROCESSED_PATH = PROCESSED_DATA_PATH + 'full_channel_processed.csv'
    FULL_CHANNEL_ENHANCED_PATH = PROCESSED_DATA_PATH + 'full_channel_enhanced.csv'

    # Ticker paths
    TICKER_RAW_17HR_PATH = RAW_DATA_PATH + 'Ticker_GDAX_20220511_17hr.json'
    TICKER_RAW_19HR_PATH = RAW_DATA_PATH + 'Ticker_GDAX_20220511_19hr.json'
    TICKER_RAW_20HR_PATH = RAW_DATA_PATH + 'Ticker_GDAX_20220511_20hr.json'
    TICKER_RAW_PATH = RAW_DATA_PATH + 'ticker.csv'
    TICKER_PROCESSED_PATH = PROCESSED_DATA_PATH + 'ticker_processed.csv'
    TICKER_ENHANCED_PATH = PROCESSED_DATA_PATH + 'ticker_enhanced.csv'

    # Data preprocessing parameters
    CATEGORICAL_COLUMNS = ['type', 'side', 'reason']
    NUMERIC_COLUMNS = ['price', 'size', 'remaining_size', 'remaining_size_change']

    # Feature engineering parameters
    ROLLING_WINDOWS = [5, 10, 15]
    OPERATIONS = ['mean', 'std', 'var']

    # Environment simulation parameters
    DEFAULT_SPOOFING_THRESHOLD = 0.8  # Default spoofing threshold for normal runs
    HISTORY_WINDOW_SIZE = 10
    FEATURE_WEIGHTS = {  # Feature weights used in anomaly score calculations
        'order_flow_imbalance': 0.3,
        'spread': 0.4,
        'cancel_to_received_ratio': 0.3
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
    