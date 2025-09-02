# MLGOptimiser Tests

This directory contains all testing files for the MLGOptimiser project.

## Directory Structure

```
tests/
├── README.md                 # This file
├── pytest.ini              # Pytest configuration
├── requirements-test.txt    # Testing dependencies
├── test_logging.py         # Original custom test suite
├── test_logging_pytest.py  # Professional pytest test suite
└── logging_example.py      # Usage examples for logging system
```

## Test Files

### `test_logging_pytest.py` (Recommended)
Professional pytest-based test suite with:
- Proper test fixtures and setup/teardown
- Parameterized tests for different scenarios
- Better error reporting and assertions
- Industry-standard test organization
- Coverage support ready

### `test_logging.py` (Legacy)
Original custom test script that provides:
- Quick validation of logging functionality
- Custom formatted output
- Simple assertion-based testing
- No external dependencies beyond standard library

### `logging_example.py`
Demonstration script showing:
- Basic logging usage examples
- Performance monitoring examples
- Error handling examples
- Different logger creation methods

## Running Tests

### Prerequisites
Install testing dependencies:
```bash
pip install -r requirements-test.txt
```

### Run pytest tests (Recommended)
```bash
# From the tests directory
pytest test_logging_pytest.py -v

# Run all tests
pytest -v

# Run with coverage
pytest --cov=../src/mlgoptimiser --cov-report=html

# Run specific test class
pytest test_logging_pytest.py::TestLoggingInitialization -v

# Run tests matching pattern
pytest -k "test_log_levels" -v
```

### Run legacy tests
```bash
# From the tests directory
python test_logging.py
```

### Run examples
```bash
# From the tests directory
python logging_example.py
```

## Test Categories

Tests are organized into categories using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.logging` - Logging-specific tests

Run specific categories:
```bash
pytest -m "unit" -v          # Run only unit tests
pytest -m "not slow" -v      # Skip slow tests
pytest -m "logging" -v       # Run only logging tests
```

## Coverage Reports

Generate coverage reports to see test coverage:
```bash
# Generate HTML coverage report
pytest --cov=../src/mlgoptimiser --cov-report=html

# Generate terminal coverage report
pytest --cov=../src/mlgoptimiser --cov-report=term-missing

# Coverage report will be in htmlcov/ directory
```

## Adding New Tests

When adding new tests:

1. **For pytest**: Follow the pattern in `test_logging_pytest.py`
   - Use descriptive test names starting with `test_`
   - Organize related tests into classes
   - Use appropriate fixtures for setup/teardown
   - Add markers for test categorization

2. **Test file naming**: Use `test_*.py` or `*_test.py`

3. **Import structure**: Use the same path setup as existing tests:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "mlgoptimiser"))
   ```

## Continuous Integration

The pytest configuration is CI/CD ready. Tests can be integrated with:
- GitHub Actions
- Jenkins
- Travis CI
- Any CI system that supports Python/pytest

Example GitHub Action workflow snippet:
```yaml
- name: Install test dependencies
  run: pip install -r tests/requirements-test.txt
  
- name: Run tests
  run: pytest tests/ -v --cov=src/mlgoptimiser
```

## Why pytest over Custom Tests?

The project now includes both approaches, but pytest is recommended because:

1. **Industry Standard**: Widely used and well-supported
2. **Better Reporting**: Clearer test results and failure messages
3. **Fixtures**: Better test setup/teardown management
4. **Parameterization**: Easy to test multiple scenarios
5. **Coverage Integration**: Built-in coverage reporting
6. **CI/CD Ready**: Easy integration with automation systems
7. **Plugin Ecosystem**: Extensive plugin support
8. **Parallel Execution**: Can run tests in parallel for speed