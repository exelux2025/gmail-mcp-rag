import logging
import sys
from typing import Optional

# Global logger instance
_logger: Optional[logging.Logger] = None


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: The name for the logger (usually __name__)
        
    Returns:
        Configured logger instance
    """
    global _logger
    
    if _logger is None:
        # Configure the root logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        _logger = logging.getLogger()
    
    return logging.getLogger(name)


def set_log_level(level: str = "INFO"):
    """
    Set the logging level for all loggers.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.getLogger().setLevel(getattr(logging, level.upper())) 