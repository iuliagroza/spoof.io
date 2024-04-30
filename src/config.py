class Config:
    RAW_DATA_PATH = '../data/raw/'
    PROCESSED_DATA_PATH = '../data/processed/'
    MODEL_SAVE_PATH = '../models/'
    REPORT_PATH = '../reports/'
    OUTPUT_PATH = '../outputs/'

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

    # Data preprocessing and feature extraction parameters
    FEATURE_WINDOW = 5  # Window size for creating rolling features
    SKIP_STEPS = 1  # Steps to skip for creating time-lagged features
    CATEGORICAL_COLUMNS = ['type', 'side', 'reason']
    NUMERIC_COLUMNS = ['price', 'size', 'remaining_size', 'remaining_size_change']

    # Training configuration
    TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% training, 20% testing
    RANDOM_SEED = 42  # Seed for any random operations to ensure reproducibility

    # Deployment configurations
    API_VERSION = 'v1'
    API_TITLE = 'spoof.io - Spoofing and Layering Detection API'
    API_DESCRIPTION = 'API for detecting spoofing and layering in trading data using PPO.'

