import logging
import sys

def setup_logger():
    """
    Sets up a standardized logger for the project.
    
    Returns:
        logging.Logger: A configured logger instance.
    """
    # Get a logger instance
    logger = logging.getLogger("CryptoLiquidityForecast")
    
    # Prevent adding handlers multiple times if the script is re-run
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create a handler to output to the console
        handler = logging.StreamHandler(sys.stdout)
        
        # Create a formatter to define the log message format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Set the formatter for the handler
        handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(handler)
        
    return logger

# Instantiate the logger so it can be imported directly
logger = setup_logger()