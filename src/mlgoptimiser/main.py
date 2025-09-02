import sys

from . import flowcontrol, input_parser
from .globals import GlobalOptimisation
from .logging_config import initialize_logging, get_auto_logger, shutdown_logging


def main():
    """Main entry point for MLGOptimiser."""
    # Initialize logging system
    initialize_logging(
        log_level="INFO",
        console_output=True,
        file_output=True
    )
    
    logger = get_auto_logger()
    logger.info("=" * 60)
    logger.info("MLGOptimiser Application Starting")
    logger.info("=" * 60)
    
    try:
        logger.info("Starting initialization phase")
        flowcontrol.initialise()
        
        logger.info("Starting execution phase")
        flowcontrol.execute()
        
        logger.info("MLGOptimiser completed successfully")
        
    except Exception as e:
        logger.critical(f"Fatal error in main application: {e}")
        logger.critical("MLGOptimiser terminated unexpectedly")
        raise
    finally:
        logger.info("Shutting down logging system")
        shutdown_logging()


if __name__ == "__main__":
    main()
