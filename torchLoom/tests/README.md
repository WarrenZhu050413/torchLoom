# torchLoom Tests

This directory contains comprehensive unit, integration, and server tests for the torchLoom package.

## Test Structure

The test suite has been reorganized and simplified into essential categories:

### Core Unit Tests
- **`test_weaver.py`**: Essential unit tests for Weaver core functionality
  - Initialization and configuration
  - Device-replica mapping logic
  - Basic message handling
  - Error handling for uninitialized state
  - Resource cleanup

### CLI Integration Tests
- **`test_cli_integration.py`**: Tests using the torchLoom CLI with real NATS servers
  - Device registration through CLI
  - Device failure simulation
  - Learning rate updates via config
  - Configuration parameter management
  - Full workflow testing (registration → failure → recovery)

### Server Integration Tests  
- **`test_server_integration.py`**: Comprehensive server tests with actual running servers
  - Weaver startup/shutdown timing and behavior
  - Full workflow testing with real servers
  - Concurrent Weaver instance testing
  - Client connection/disconnection resilience
  - Performance testing under load (50+ messages)

### Real NATS Integration Tests
- **`test_integration_real_nats.py`**: Integration tests using real NATS servers
  - End-to-end message flow testing
  - Config info handling with learning rate extraction
  - Subscription management verification

### Additional Component Tests
- **`test_threadlet_callback.py`**: Threadlet callback system tests
  - Enhanced handler system with decorators
  - Type validation and conversion
  - Lightning integration testing

### Test Utilities
- **`test_utils.py`**: Common testing utilities
  - `NatsTestServer`: Manages real nats-server instances for testing
  - Message collection helpers
  - **`conftest.py`**: Pytest fixtures for mocks and test setup

## Test Coverage Analysis

### What's Well Covered ✅
1. **Core Weaver Functionality**: Device mapping, initialization, cleanup
2. **CLI Message Flow**: All message types (register, fail, config) work through real NATS
3. **Server Integration**: Startup/shutdown, concurrent instances, client resilience
4. **Real NATS Integration**: Actual message flow through real servers
5. **Performance**: Load testing with timing assertions
6. **Error Handling**: Uninitialized state errors, connection failures

### Areas for Improvement 📈
1. **JetStream Testing**: Current focus is on regular NATS, JetStream usage is minimal
2. **Threadlet Integration**: More comprehensive threadlet-weaver interaction tests
3. **Failure Recovery**: More sophisticated failure scenarios and recovery patterns
4. **Configuration Validation**: Edge cases in config parameter handling
5. **Network Resilience**: NATS server restart/reconnection scenarios

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
# Core unit tests (fast)
pytest tests/test_weaver.py -v

# CLI integration tests with real NATS
pytest tests/test_cli_integration.py -v

# Server integration tests (comprehensive)
pytest tests/test_server_integration.py -v

# Real NATS integration tests
pytest tests/test_integration_real_nats.py -v
```

### Run with Coverage
```bash
pytest --cov=torchLoom --cov-report=html
```

### Run with Debug Output
```bash
pytest -v -s  # Shows print statements and detailed output
```

## Test Philosophy

The tests follow a **minimal mocking, maximum realism** approach:

1. **Unit Tests**: Test core logic with minimal dependencies
2. **Integration Tests**: Use real NATS servers and CLI clients
3. **Server Tests**: Start actual server processes to test realistic scenarios
4. **Performance Tests**: Include timing assertions to catch regressions

This approach provides higher confidence that the system works in real-world scenarios while maintaining fast feedback for core functionality.

## Adding New Tests

When adding new functionality:

1. **Start with unit tests** in `test_weaver.py` for core logic
2. **Add CLI integration tests** in `test_cli_integration.py` for new CLI commands
3. **Add server tests** in `test_server_integration.py` for complex workflows
4. **Include timing assertions** for performance-critical paths
5. **Use real servers** rather than mocks when testing message flow 