# DEPENDENCIES
import sys
import logging
import warnings
from config.settings import LOG_FILE
from config.settings import LOG_LEVEL
from config.settings import LOG_FORMAT


# Filter out all the warnings
warnings.filterwarnings("ignore")


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Setup and configure logger with file and console handlers
    
    Arguments:
    ----------
        name   { str }     : Name of the logger
        
    Returns:
    --------
        { logging.Logger } : Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # File handler
    file_handler      = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)

    file_formatter    = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler   = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger