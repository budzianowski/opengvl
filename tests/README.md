# OpenGVL Test Suite

This directory contains comprehensive pytest tests for all core functionalities of the OpenGVL codebase.

## Test Coverage

The test suite includes **88 tests** covering the following modules:

### Core Data Structures (`test_data_types.py`)
- **Episode** - Core episode data structure with validation
- **InferredEpisode** - Episode with model predictions
- **Example** - Few-shot examples with context
- **InferredFewShotResult** - Complete inference results

### Metrics (`test_voc_metric.py`)
- **VOCMetric** - Value-Order Correlation metric computation
- Edge cases: constant predictions, insufficient data, empty episodes
- Perfect positive/negative correlations
- Partial correlations and random ordering

### Error Handling (`test_errors.py`)
- All custom exception classes
- Error message formatting
- Error inheritance and chaining
- Common error usage patterns

### Data Loaders (`test_data_loaders.py`)
- **BaseDataLoader** abstract class functionality
- Linear completion rate generation
- Frame shuffling and data integrity
- Random number generation with seeds
- Concrete loader implementation patterns

### Inference Utilities (`test_inference_core.py`)
- Percentage extraction from text with regex
- Decimal handling and normalization
- Count validation and error handling
- JSONL serialization
- Inferred example building

### Prediction Results (`test_prediction_results.py`)
- **PredictionRecord** serialization and deserialization
- **DatasetMetrics** aggregation
- Metric computation across multiple records
- Handling of None and non-numeric values
- Batch processing workflows

### Utilities and Integration (`test_utilities.py`)
- Constants validation
- Type alias functionality
- End-to-end data flow testing
- Error propagation patterns
- Serialization round-trips

## Running Tests

### Basic Commands
```bash
# Run all tests
make test

# Run tests quickly (quiet mode)
make test-fast

# Run tests with coverage report
make test-coverage

# Run tests with extra verbose output
make test-verbose
```

### Specific Test Commands
```bash
# Run specific test file
make test-specific TEST=test_voc_metric.py

# Run specific test class
make test-specific TEST="test_data_types.py::TestEpisode"

# Run specific test method
make test-specific TEST="test_errors.py::TestImageEncodingError::test_default_message"
```

### Advanced Commands
```bash
# Generate CI-friendly reports
make test-ci

# Run tests in parallel (requires pytest-xdist)
make test-parallel
```

## Test Organization

Tests are organized by module and functionality:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component interactions
- **Edge Case Tests**: Boundary conditions and error scenarios
- **Pattern Tests**: Common usage patterns and workflows

## Coverage Report

The test suite achieves comprehensive coverage of core functionalities:

- **Data Types**: 100% coverage
- **Error Classes**: 100% coverage  
- **Prediction Results**: 100% coverage
- **VOC Metrics**: 96% coverage
- **Constants/Aliases**: 100% coverage

Areas with limited coverage (clients, scripts) are due to external dependencies (APIs, models) that require mocking or are not part of core functionality testing.

## Configuration

Test configuration is managed through:

- `pytest.ini` - Pytest settings and markers
- `Makefile` - Test execution commands
- Coverage settings in pytest commands

## Writing New Tests

When adding new functionality:

1. Create test files following the `test_*.py` naming convention
2. Organize tests in classes with descriptive names (`TestClassName`)
3. Use descriptive test method names (`test_specific_functionality`)
4. Include edge cases and error scenarios
5. Add integration tests for cross-module functionality
6. Update this README with new test coverage

## Test Dependencies

The test suite uses:
- `pytest` - Test framework
- `pytest-cov` - Coverage reporting
- `numpy` - Array operations in tests
- `unittest.mock` - Mocking external dependencies

All test dependencies are available in the standard Python environment and don't require external API keys or model downloads.