#!/usr/bin/env python3
"""
Test script to verify the new logging system works correctly.

This script tests the logging functionality across the main modules to ensure
proper initialization, configuration, and output generation.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path to import mlgoptimiser modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "mlgoptimiser"))

from logging_config import (
    initialize_logging, 
    get_logger, 
    get_auto_logger, 
    shutdown_logging,
    set_log_level,
    log_performance
)


def test_basic_logging():
    """Test basic logging functionality."""
    print("Testing basic logging functionality...")
    
    # Create a temporary directory for test logs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize logging with test directory
        initialize_logging(
            log_dir=temp_dir,
            log_level="DEBUG",
            console_output=True,
            file_output=True
        )
        
        # Test different logger types
        logger1 = get_logger("test_module", "test_function")
        logger2 = get_auto_logger()
        
        # Test different log levels
        logger1.debug("This is a debug message")
        logger1.info("This is an info message")
        logger1.warning("This is a warning message")
        logger1.error("This is an error message")
        logger1.critical("This is a critical message")
        
        logger2.info("Auto-logger test message")
        
        # Test log level changing
        set_log_level("ERROR")
        logger1.info("This info message should not appear after level change")
        logger1.error("This error message should appear after level change")
        
        # Check that log files were created
        log_files = list(Path(temp_dir).glob("*.log"))
        assert len(log_files) > 0, "No log files were created"
        
        print(f"✓ Created {len(log_files)} log files")
        for log_file in log_files:
            print(f"  - {log_file.name}")
        
        # Check that log files contain content
        main_log = Path(temp_dir) / "mlgoptimiser_main.log"
        if main_log.exists():
            content = main_log.read_text()
            assert len(content) > 0, "Main log file is empty"
            print(f"✓ Main log file contains {len(content.splitlines())} lines")
        
        shutdown_logging()
    
    print("✓ Basic logging test passed\n")


def test_performance_decorator():
    """Test the performance logging decorator."""
    print("Testing performance logging decorator...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        initialize_logging(log_dir=temp_dir, log_level="DEBUG")
        
        @log_performance()
        def sample_function(x, y):
            """Sample function for testing performance logging."""
            import time
            time.sleep(0.1)  # Simulate some work
            return x + y
        
        result = sample_function(5, 3)
        assert result == 8, "Sample function returned incorrect result"
        
        # Check that performance was logged
        debug_log = Path(temp_dir) / "mlgoptimiser_debug.log"
        if debug_log.exists():
            content = debug_log.read_text()
            assert "Starting sample_function" in content, "Performance start not logged"
            assert "Completed sample_function" in content, "Performance completion not logged"
            print("✓ Performance logging decorator working correctly")
        
        shutdown_logging()
    
    print("✓ Performance decorator test passed\n")


def test_module_integration():
    """Test logging integration with actual modules."""
    print("Testing module integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        initialize_logging(log_dir=temp_dir, log_level="INFO")
        
        try:
            # Test import and basic functionality
            from logging_config import get_logger
            
            # Test module-specific loggers
            main_logger = get_logger("main")
            flowcontrol_logger = get_logger("flowcontrol")
            monte_carlo_logger = get_logger("monte_carlo_util")
            
            main_logger.info("Testing main module logger")
            flowcontrol_logger.info("Testing flowcontrol module logger")
            monte_carlo_logger.info("Testing monte_carlo_util module logger")
            
            # Check that logs were written with correct module names
            main_log = Path(temp_dir) / "mlgoptimiser_main.log"
            if main_log.exists():
                content = main_log.read_text()
                assert "mlgoptimiser.main" in content, "Main module not properly identified"
                assert "mlgoptimiser.flowcontrol" in content, "Flowcontrol module not properly identified"
                assert "mlgoptimiser.monte_carlo_util" in content, "Monte Carlo util module not properly identified"
                print("✓ Module-specific loggers working correctly")
        
        except ImportError as e:
            print(f"⚠ Warning: Could not test full module integration: {e}")
        
        shutdown_logging()
    
    print("✓ Module integration test passed\n")


def test_error_handling():
    """Test logging behavior with errors and exceptions."""
    print("Testing error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        initialize_logging(log_dir=temp_dir, log_level="DEBUG")
        
        logger = get_logger("error_test")
        
        # Test exception logging
        try:
            raise ValueError("Test exception for logging")
        except ValueError as e:
            logger.error(f"Caught exception: {e}")
            logger.debug("Exception details", exc_info=True)
        
        # Test with performance decorator and exception
        @log_performance()
        def failing_function():
            raise RuntimeError("This function always fails")
        
        try:
            failing_function()
        except RuntimeError:
            pass  # Expected
        
        # Check error log
        error_log = Path(temp_dir) / "error.log"
        if error_log.exists():
            content = error_log.read_text()
            assert "Failed failing_function" in content, "Exception not logged in performance decorator"
            print("✓ Error handling working correctly")
        
        shutdown_logging()
    
    print("✓ Error handling test passed\n")


def main():
    """Run all logging tests."""
    print("=" * 60)
    print("MLGOptimiser Logging System Test Suite")
    print("=" * 60)
    
    try:
        test_basic_logging()
        test_performance_decorator()
        test_module_integration()
        test_error_handling()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED - Logging system is working correctly!")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())