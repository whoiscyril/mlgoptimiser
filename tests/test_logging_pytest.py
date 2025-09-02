"""
Proper pytest-based test suite for the MLGOptimiser logging system.

This replaces the custom test script with industry-standard pytest tests
that provide better organization, reporting, and integration capabilities.
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Add the src directory to the path to import mlgoptimiser modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "mlgoptimiser"))

from logging_config import (
    initialize_logging,
    get_logger,
    get_auto_logger,
    shutdown_logging,
    set_log_level,
    log_performance,
    MLGLogger
)


@pytest.fixture
def temp_log_dir():
    """Provide a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def clean_logger_state():
    """Ensure clean logger state before and after each test."""
    # Reset logger state before test
    MLGLogger._initialized = False
    MLGLogger._loggers.clear()
    
    yield
    
    # Clean up after test
    try:
        shutdown_logging()
    except:
        pass
    MLGLogger._initialized = False
    MLGLogger._loggers.clear()


class TestLoggingInitialization:
    """Test logging system initialization and configuration."""

    def test_initialize_logging_creates_log_files(self, temp_log_dir, clean_logger_state):
        """Test that initialization creates the expected log files."""
        initialize_logging(
            log_dir=temp_log_dir,
            log_level="DEBUG",
            console_output=False,
            file_output=True
        )
        
        logger = get_logger("test")
        logger.info("Test message")
        
        # Check that expected log files are created
        expected_files = [
            "mlgoptimiser_main.log",
            "mlgoptimiser_debug.log", 
            "error.log",
            f"simulation_{time.strftime('%Y%m%d')}.log"
        ]
        
        for filename in expected_files:
            log_file = Path(temp_log_dir) / filename
            assert log_file.exists(), f"Expected log file {filename} not created"
            
    def test_initialize_logging_console_only(self, clean_logger_state):
        """Test initialization with console output only."""
        initialize_logging(
            console_output=True,
            file_output=False
        )
        
        logger = get_logger("test")
        # Should not raise exception
        logger.info("Console test message")
        
    def test_initialize_logging_multiple_calls(self, temp_log_dir, clean_logger_state):
        """Test that multiple initialization calls don't cause issues."""
        initialize_logging(log_dir=temp_log_dir)
        initialize_logging(log_dir=temp_log_dir)  # Should not reinitialize
        
        logger = get_logger("test")
        logger.info("Multiple init test")
        
        # Should still work normally
        assert MLGLogger._initialized


class TestLoggerCreation:
    """Test different ways of creating loggers."""
    
    def test_get_logger_with_module_and_function(self, temp_log_dir, clean_logger_state):
        """Test creating logger with explicit module and function names."""
        initialize_logging(log_dir=temp_log_dir)
        
        logger = get_logger("test_module", "test_function")
        logger.info("Test message")
        
        # Check that logger name is correct
        assert logger.name == "mlgoptimiser.test_module.test_function"
        
    def test_get_logger_module_only(self, temp_log_dir, clean_logger_state):
        """Test creating logger with module name only."""
        initialize_logging(log_dir=temp_log_dir)
        
        logger = get_logger("test_module")
        logger.info("Test message")
        
        assert logger.name == "mlgoptimiser.test_module"
        
    def test_get_auto_logger(self, temp_log_dir, clean_logger_state):
        """Test automatic logger detection."""
        initialize_logging(log_dir=temp_log_dir)
        
        logger = get_auto_logger()
        logger.info("Auto logger test")
        
        # Should detect this test function
        assert "test_get_auto_logger" in logger.name
        
    def test_logger_caching(self, temp_log_dir, clean_logger_state):
        """Test that loggers are cached and reused."""
        initialize_logging(log_dir=temp_log_dir)
        
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")
        
        assert logger1 is logger2, "Loggers should be cached and reused"


class TestLogLevels:
    """Test different log levels and filtering."""
    
    @pytest.mark.parametrize("log_level,should_appear", [
        ("DEBUG", {"debug": True, "info": True, "warning": True, "error": True}),
        ("INFO", {"debug": False, "info": True, "warning": True, "error": True}),
        ("WARNING", {"debug": False, "info": False, "warning": True, "error": True}),
        ("ERROR", {"debug": False, "info": False, "warning": False, "error": True}),
    ])
    def test_log_levels(self, temp_log_dir, clean_logger_state, log_level, should_appear):
        """Test that log level filtering works correctly."""
        initialize_logging(
            log_dir=temp_log_dir,
            log_level=log_level,
            console_output=False
        )
        
        logger = get_logger("test")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Check main log file content
        main_log = Path(temp_log_dir) / "mlgoptimiser_main.log"
        content = main_log.read_text()
        
        if should_appear["debug"]:
            assert "Debug message" in content
        else:
            assert "Debug message" not in content
            
        if should_appear["info"]:
            assert "Info message" in content
        else:
            assert "Info message" not in content
            
        if should_appear["warning"]:
            assert "Warning message" in content
        else:
            assert "Warning message" not in content
            
        if should_appear["error"]:
            assert "Error message" in content
        else:
            assert "Error message" not in content
    
    def test_dynamic_log_level_change(self, temp_log_dir, clean_logger_state):
        """Test changing log level at runtime."""
        initialize_logging(log_dir=temp_log_dir, log_level="DEBUG")
        
        logger = get_logger("test")
        logger.debug("Debug before change")
        
        set_log_level("ERROR")
        logger.debug("Debug after change")
        logger.error("Error after change")
        
        main_log = Path(temp_log_dir) / "mlgoptimiser_main.log"
        content = main_log.read_text()
        
        assert "Debug before change" in content
        assert "Debug after change" not in content
        assert "Error after change" in content


class TestPerformanceLogging:
    """Test performance logging decorator."""
    
    def test_performance_decorator_success(self, temp_log_dir, clean_logger_state):
        """Test performance decorator with successful function."""
        initialize_logging(log_dir=temp_log_dir, log_level="DEBUG")
        
        @log_performance()
        def test_function(x, y):
            time.sleep(0.01)  # Small delay
            return x + y
            
        result = test_function(5, 3)
        assert result == 8
        
        # Check that performance was logged
        debug_log = Path(temp_log_dir) / "mlgoptimiser_debug.log"
        content = debug_log.read_text()
        
        assert "Starting test_function" in content
        assert "Completed test_function" in content
        assert "args=2, kwargs=0" in content
        
    def test_performance_decorator_exception(self, temp_log_dir, clean_logger_state):
        """Test performance decorator with function that raises exception."""
        initialize_logging(log_dir=temp_log_dir, log_level="DEBUG")
        
        @log_performance()
        def failing_function():
            raise ValueError("Test exception")
            
        with pytest.raises(ValueError):
            failing_function()
        
        # Check that failure was logged
        debug_log = Path(temp_log_dir) / "mlgoptimiser_debug.log"
        content = debug_log.read_text()
        
        assert "Starting failing_function" in content
        assert "Failed failing_function" in content
        assert "Test exception" in content


class TestErrorLogging:
    """Test error logging and exception handling."""
    
    def test_error_file_separation(self, temp_log_dir, clean_logger_state):
        """Test that errors are written to separate error file."""
        initialize_logging(log_dir=temp_log_dir)
        
        logger = get_logger("test")
        logger.info("Info message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        # Check error file contains only errors
        error_log = Path(temp_log_dir) / "error.log"
        error_content = error_log.read_text()
        
        assert "Info message" not in error_content
        assert "Error message" in error_content
        assert "Critical message" in error_content
        
    def test_exception_logging_with_traceback(self, temp_log_dir, clean_logger_state):
        """Test logging exceptions with traceback."""
        initialize_logging(log_dir=temp_log_dir, log_level="DEBUG")
        
        logger = get_logger("test")
        
        try:
            raise ZeroDivisionError("Test division by zero")
        except ZeroDivisionError:
            logger.error("Exception occurred", exc_info=True)
            
        debug_log = Path(temp_log_dir) / "mlgoptimiser_debug.log"
        content = debug_log.read_text()
        
        assert "Exception occurred" in content
        assert "ZeroDivisionError" in content
        assert "Test division by zero" in content


class TestLogFileRotation:
    """Test log file rotation and management."""
    
    def test_log_file_rotation_on_size(self, temp_log_dir, clean_logger_state):
        """Test that log files rotate when they get too large."""
        # This is a more complex test that would require generating large logs
        # For now, just test that rotation is configured
        initialize_logging(log_dir=temp_log_dir)
        
        logger = get_logger("test")
        logger.info("Test message")
        
        # Verify that handlers are configured with rotation
        root_logger = logger.parent
        rotating_handlers = [
            h for h in root_logger.handlers 
            if hasattr(h, 'maxBytes') and h.maxBytes > 0
        ]
        
        assert len(rotating_handlers) > 0, "No rotating file handlers configured"


class TestModuleIntegration:
    """Test integration with actual MLGOptimiser modules."""
    
    def test_module_logger_naming(self, temp_log_dir, clean_logger_state):
        """Test that module loggers have correct names."""
        initialize_logging(log_dir=temp_log_dir)
        
        main_logger = get_logger("main")
        flowcontrol_logger = get_logger("flowcontrol")
        monte_carlo_logger = get_logger("monte_carlo_util")
        
        assert main_logger.name == "mlgoptimiser.main"
        assert flowcontrol_logger.name == "mlgoptimiser.flowcontrol"
        assert monte_carlo_logger.name == "mlgoptimiser.monte_carlo_util"
        
    def test_log_message_format(self, temp_log_dir, clean_logger_state):
        """Test that log messages have correct format."""
        initialize_logging(log_dir=temp_log_dir, console_output=False)
        
        logger = get_logger("test_module", "test_function")
        logger.info("Test formatting message")
        
        # Force flush all handlers
        for handler in logger.handlers:
            handler.flush()
        for handler in logger.parent.handlers:
            handler.flush()
        
        main_log = Path(temp_log_dir) / "mlgoptimiser_main.log"
        content = main_log.read_text()
        
        # Check format: [TIMESTAMP] [LEVEL] [MODULE.FUNCTION] MESSAGE
        lines = [line for line in content.split('\n') if 'Test formatting message' in line]
        assert len(lines) >= 1, f"Expected to find test message in log content: {content}"
        
        line = lines[0]
        assert '[INFO]' in line
        assert '[mlgoptimiser.test_module.test_function]' in line
        assert 'Test formatting message' in line
        
        # Basic timestamp format check (YYYY-MM-DD HH:MM:SS)
        import re
        timestamp_pattern = r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]'
        assert re.search(timestamp_pattern, line), f"Invalid timestamp format in: {line}"


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])