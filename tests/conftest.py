"""
Shared pytest fixtures for MLGOptimiser tests.

This file provides common fixtures that can be used across multiple test files.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add the src directory to the path to import mlgoptimiser modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "mlgoptimiser"))

from logging_config import MLGLogger, shutdown_logging


@pytest.fixture
def temp_log_dir():
    """
    Provide a temporary directory for log files.
    
    This fixture creates a temporary directory that will be automatically
    cleaned up after the test completes.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def clean_logger_state():
    """
    Ensure clean logger state before and after each test.
    
    This fixture resets the global logger state to ensure tests don't
    interfere with each other.
    """
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


@pytest.fixture
def sample_log_messages():
    """
    Provide sample log messages for testing.
    
    Returns a dictionary with different types of log messages
    that can be used in tests.
    """
    return {
        "debug": "This is a debug message for testing",
        "info": "This is an info message for testing", 
        "warning": "This is a warning message for testing",
        "error": "This is an error message for testing",
        "critical": "This is a critical message for testing"
    }


@pytest.fixture
def mock_time_sleep(monkeypatch):
    """
    Mock time.sleep to speed up tests that use delays.
    
    This prevents tests from actually sleeping, making them run faster.
    """
    def mock_sleep(duration):
        pass
    
    monkeypatch.setattr("time.sleep", mock_sleep)


# Test markers for categorizing tests
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests") 
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "logging: Logging system tests")


def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on their names and locations.
    
    This adds appropriate markers to tests automatically.
    """
    for item in items:
        # Mark logging tests
        if "logging" in item.nodeid.lower():
            item.add_marker(pytest.mark.logging)
        
        # Mark slow tests (those that use sleep or have 'slow' in name)
        if "slow" in item.name.lower() or "sleep" in str(item.function):
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        else:
            # Default to unit tests
            item.add_marker(pytest.mark.unit)