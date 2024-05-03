import logging
from src.config import Config

def setup_logger(filepath=None):
    """
    Sets up a logger with a specified configuration. Logs to file if is_in_file is True.
    
    Args:
        filepath (str): Path to the output log file.

    Returns:
        logging.Logger: A logger object configured with either file and stream handlers or only stream handler based on is_in_file flag.
    """
    logger = logging.getLogger(__name__)

    if not logger.handlers:
        logger.setLevel(Config.LOG_LEVEL)
        formatter = logging.Formatter(Config.LOG_FORMAT)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if filepath:
        formatter = logging.Formatter(Config.LOG_FORMAT)
        file_handler = logging.FileHandler(filepath, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
