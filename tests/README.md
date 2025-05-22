# torchLoom Tests

This directory contains unit and integration tests for the torchLoom package.

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_weaver.py
```

To run tests with increased verbosity:

```bash
pytest -v
```

To run tests with code coverage:

```bash
pytest --cov=torchLoom
```

To generate an HTML coverage report:

```bash
pytest --cov=torchLoom --cov-report=html
```

## Test Structure

The test suite is organized as follows:

- `conftest.py`: Contains pytest fixtures used across multiple test files
- `test_config.py`: Tests for the configuration module
- `test_constants.py`: Tests for constants and subject definitions
- `test_integration.py`: Integration tests that test multiple components working together
- `test_logging.py`: Tests for the logging utilities
- `test_monitor_cli.py`: Tests for the command-line interface
- `test_proto.py`: Tests for protocol buffer message handling
- `test_utils.py`: Tests for utility functions
- `test_weaver.py`: Tests for the Weaver class, which is the central component of torchLoom

## Test Coverage

The tests cover:

1. Basic functionality of each module
2. Edge cases and error handling
3. Integration between components
4. Protocol buffer message serialization and deserialization
5. Device-replica mapping operations
6. Event handling and propagation
7. Command-line interface functionality

## Adding New Tests

When adding new functionality to torchLoom, please add corresponding tests following these guidelines:

1. Create unit tests for each new function or method
2. Update integration tests if the new functionality affects multiple components
3. Mock external dependencies (like NATS) to avoid actual network operations during testing
4. Include both success and failure cases in your tests
5. Use descriptive test names that explain what is being tested 