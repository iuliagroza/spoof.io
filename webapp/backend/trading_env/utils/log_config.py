import logging
from trading_env.config import Config


def setup_logger(name, filepath=None, file_log_level=None):
    """
    Sets up a logger with a specified configuration. Logs to file if filepath is provided and controls file logging level if specified.
    
    Args:
        name (str): The name of the logger, typically __name__ from the caller.
        filepath (str): Optional. Path to the output log file.
        file_log_level (int): Optional. Specific log level for file logging.

    Returns:
        logging.Logger: A logger object configured with either file and stream handlers or only stream handler based on filepath.
    """
    # Create a logger with the specified name to prevent overlap with other loggers
    logger = logging.getLogger(name)

    # Prevent logging propagation to root logger which might duplicate logs in other handlers
    logger.propagate = False

    logger.setLevel(Config.LOG_LEVEL)
    formatter = logging.Formatter(Config.LOG_FORMAT)

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if filepath and not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        file_handler = logging.FileHandler(filepath, mode='w')
        file_handler.setFormatter(formatter)
        if file_log_level:
            file_handler.setLevel(file_log_level)
        logger.addHandler(file_handler)

    return logger
