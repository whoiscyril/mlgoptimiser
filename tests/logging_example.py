#!/usr/bin/env python3
"""
Example usage of the MLGOptimiser logging system.

This demonstrates how to use the new centralized logging system
in different modules and scenarios.
"""

import sys
from pathlib import Path

# Add the src directory to the path to import mlgoptimiser modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "mlgoptimiser"))

from logging_config import (
    initialize_logging, 
    get_logger, 
    get_auto_logger,
    log_performance,
    set_log_level,
    shutdown_logging
)


def demonstrate_basic_usage():
    """Show basic logging usage."""
    # Get a logger for this specific module and function
    logger = get_logger("example", "demonstrate_basic_usage")
    
    logger.info("This is how you log an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message (may not appear depending on log level)")


def demonstrate_auto_logger():
    """Show automatic logger detection."""
    # Automatically detects module and function name
    logger = get_auto_logger()
    
    logger.info("This message will show the correct module and function automatically")


@log_performance()
def demonstrate_performance_logging():
    """Show performance logging with decorator."""
    import time
    
    logger = get_auto_logger()
    logger.info("Doing some work...")
    
    # Simulate some work
    time.sleep(0.5)
    
    logger.info("Work completed")
    return "Task finished"


def demonstrate_error_logging():
    """Show how to log exceptions."""
    logger = get_auto_logger()
    
    try:
        # Simulate an error
        result = 10 / 0
    except ZeroDivisionError as e:
        logger.error(f"Mathematical error occurred: {e}")
        logger.debug("Full exception details:", exc_info=True)


def main():
    """Main demonstration function."""
    print("MLGOptimiser Logging System - Usage Example")
    print("=" * 50)
    
    # Initialize the logging system
    # In real usage, this would be done in main.py
    initialize_logging(
        log_level="DEBUG",
        console_output=True,
        file_output=True
    )
    
    logger = get_auto_logger()
    logger.info("Starting logging demonstration")
    
    # Demonstrate different logging features
    demonstrate_basic_usage()
    demonstrate_auto_logger()
    result = demonstrate_performance_logging()
    logger.info(f"Performance demo returned: {result}")
    demonstrate_error_logging()
    
    # Show how to change log level dynamically
    logger.info("Changing log level to WARNING - debug messages will no longer appear")
    set_log_level("WARNING")
    
    logger.debug("This debug message should not appear")
    logger.warning("This warning message should appear")
    
    logger.warning("Demonstration completed - check the logs directory for output files")
    
    # Clean shutdown
    shutdown_logging()


if __name__ == "__main__":
    main()