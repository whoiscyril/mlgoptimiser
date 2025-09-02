"""
Central logging configuration module for MLGOptimiser.

This module provides a unified logging system that can be used across all modules
in the MLGOptimiser package. It supports multiple output destinations, configurable
log levels, and automatic module/function detection.
"""

import inspect
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class MLGLogger:
    """Custom logger wrapper with enhanced functionality for MLGOptimiser."""
    
    _loggers = {}
    _initialized = False
    _log_dir = None
    
    @classmethod
    def initialize(cls, log_dir: Optional[str] = None, log_level: str = "INFO", 
                   console_output: bool = True, file_output: bool = True):
        """
        Initialize the logging system.
        
        Args:
            log_dir: Directory for log files. If None, creates 'logs' in current directory
            log_level: Global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Whether to output logs to console
            file_output: Whether to output logs to files
        """
        if cls._initialized:
            return
            
        # Set up log directory
        if log_dir is None:
            cls._log_dir = Path.cwd() / "logs"
        else:
            cls._log_dir = Path(log_dir)
        
        cls._log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger('mlgoptimiser')
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Define log format
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handlers
        if file_output:
            # Main log file
            main_log = cls._log_dir / "mlgoptimiser_main.log"
            file_handler = logging.handlers.RotatingFileHandler(
                main_log, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            # Debug log file (captures all levels)
            debug_log = cls._log_dir / "mlgoptimiser_debug.log"
            debug_handler = logging.handlers.RotatingFileHandler(
                debug_log, maxBytes=10*1024*1024, backupCount=3
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(formatter)
            root_logger.addHandler(debug_handler)
            
            # Error-only log file
            error_log = cls._log_dir / "error.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log, maxBytes=5*1024*1024, backupCount=3
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)
            
            # Daily simulation log
            today = datetime.now().strftime('%Y%m%d')
            sim_log = cls._log_dir / f"simulation_{today}.log"
            sim_handler = logging.FileHandler(sim_log)
            sim_handler.setFormatter(formatter)
            root_logger.addHandler(sim_handler)
        
        cls._initialized = True
        root_logger.info("MLGOptimiser logging system initialized")
    
    @classmethod
    def get_logger(cls, module_name: str, function_name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance for a specific module and optionally function.
        
        Args:
            module_name: Name of the module (e.g., 'main', 'flowcontrol')
            function_name: Optional function name for more specific logging
            
        Returns:
            Logger instance configured for the module/function
        """
        if not cls._initialized:
            cls.initialize()
        
        logger_name = f"mlgoptimiser.{module_name}"
        if function_name:
            logger_name += f".{function_name}"
            
        if logger_name not in cls._loggers:
            logger = logging.getLogger(logger_name)
            cls._loggers[logger_name] = logger
        
        return cls._loggers[logger_name]
    
    @classmethod  
    def get_auto_logger(cls, depth: int = 1) -> logging.Logger:
        """
        Automatically detect the calling module and function for logger creation.
        
        Args:
            depth: How many stack frames to go up (1 = direct caller)
            
        Returns:
            Logger instance for the calling module/function
        """
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the caller
            for _ in range(depth):
                frame = frame.f_back
                
            module_name = frame.f_globals.get('__name__', 'unknown')
            function_name = frame.f_code.co_name
            
            # Clean up module name (remove package prefix)
            if module_name.startswith('mlgoptimiser.'):
                module_name = module_name.replace('mlgoptimiser.', '')
            elif module_name == '__main__':
                module_name = 'main'
            
            return cls.get_logger(module_name, function_name)
        finally:
            del frame
    
    @classmethod
    def set_level(cls, level: Union[str, int]):
        """
        Change the global log level at runtime.
        
        Args:
            level: New log level (string or logging constant)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
            
        root_logger = logging.getLogger('mlgoptimiser')
        root_logger.setLevel(level)
        
        for logger in cls._loggers.values():
            logger.setLevel(level)
    
    @classmethod
    def shutdown(cls):
        """Shutdown the logging system and close all handlers."""
        root_logger = logging.getLogger('mlgoptimiser')
        for handler in root_logger.handlers:
            handler.close()
        logging.shutdown()
        cls._initialized = False
        cls._loggers.clear()


# Convenience functions for easy usage
def get_logger(module_name: str, function_name: Optional[str] = None) -> logging.Logger:
    """Get a logger for a specific module/function."""
    return MLGLogger.get_logger(module_name, function_name)


def get_auto_logger() -> logging.Logger:
    """Automatically detect caller and return appropriate logger."""
    return MLGLogger.get_auto_logger(depth=2)  # +1 for this function call


def initialize_logging(**kwargs):
    """Initialize the logging system with optional parameters."""
    MLGLogger.initialize(**kwargs)


def set_log_level(level: Union[str, int]):
    """Set the global log level."""
    MLGLogger.set_level(level)


def shutdown_logging():
    """Shutdown the logging system."""
    MLGLogger.shutdown()


# Performance timing decorator
def log_performance(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Optional specific logger to use
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if logger is None:
                log = get_auto_logger()
            else:
                log = logger
                
            start_time = datetime.now()
            log.debug(f"Starting {func.__name__} with args={len(args)}, kwargs={len(kwargs)}")
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                log.info(f"Completed {func.__name__} in {duration:.3f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                log.error(f"Failed {func.__name__} after {duration:.3f}s: {e}")
                raise
                
        return wrapper
    return decorator