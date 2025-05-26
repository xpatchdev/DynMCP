# AutoMCP Test Suite

This directory contains comprehensive test scripts for the AutoMCP class, which automatically exposes class methods as MCP (Model Context Protocol) endpoints.

## Files

### Core Files
- `auto_mcp.py` - The main AutoMCP class implementation
- `test_auto_mcp.py` - Comprehensive test suite for AutoMCP functionality
- `run_auto_mcp_tests.py` - Interactive test runner
- `test_auto_mcp_client.py` - Client test script for testing running servers

### Test Documentation
- `AUTO_MCP_TESTS_README.md` - This file

## Quick Start

### 1. Run the Basic Test Suite
```bash
python3 test_auto_mcp.py
```

This runs a comprehensive test suite that verifies:
- Basic method registration
- Method filtering and exclusion
- Complex method signatures
- Signature analysis
- Method wrapper functionality
- Async method support
- Built-in info endpoints

### 2. Interactive Test Mode
```bash
python3 test_auto_mcp.py --interactive
```

This mode creates an AutoMCP server instance and shows you all registered methods without actually starting the server.

### 3. Start a Test Server
```bash
python3 test_auto_mcp.py --run-server --port 8000
```

This starts an actual AutoMCP server with test methods on the specified port.

### 4. Test a Running Server
```bash
# In another terminal, while the server is running:
python3 test_auto_mcp_client.py http://localhost:8000
```

This tests the actual HTTP endpoints of a running AutoMCP server.

### 5. Interactive Test Runner
```bash
python3 run_auto_mcp_tests.py
```

This provides an interactive menu to run different types of tests.

## Test Classes

The test suite includes several test classes to verify different aspects of AutoMCP:

### TestCalculator
Basic arithmetic operations with simple signatures:
- `add(a: int, b: int) -> int`
- `subtract(a: int, b: int) -> int`
- `multiply(a: float, b: float) -> float`
- `divide(a: float, b: float) -> float` (includes error handling)
- `_private_method()` (should not be exposed)

### TestStringProcessor
String manipulation methods:
- `uppercase(text: str) -> str`
- `lowercase(text: str) -> str`
- `reverse_string(text: str) -> str`
- `count_words(text: str) -> int`
- `repeat_text(text: str, times: int = 1) -> str` (includes default parameter)

### TestDataProcessor
Complex data processing with advanced types:
- `process_list(items: List[int]) -> Dict[str, Any]`
- `filter_even(numbers: List[int]) -> List[int]`
- `create_user_profile(name: str, age: int, email: Optional[str] = None) -> Dict[str, Any]`

### TestAsyncOperations
Testing async method support:
- `async_wait(seconds: float) -> str` (async method)
- `sync_method(message: str) -> str` (regular sync method)

## Test Coverage

The test suite verifies the following functionality:

### ✅ Basic Registration
- Public methods are automatically registered
- Private methods (starting with `_`) are excluded
- Magic methods (`__method__`) are excluded
- Callable attributes are registered, non-callable are ignored

### ✅ Method Filtering
- `method_filter` parameter to include only specific methods
- `exclude_methods` parameter to exclude specific methods
- `include_private` parameter to include private methods

### ✅ Signature Analysis
- Parameter detection and type hints
- Required vs optional parameters
- Default values
- Return type annotation
- Docstring extraction

### ✅ Method Wrapping
- Proper async/sync handling
- JSON serialization of results
- Error handling and structured error responses
- Preservation of original method behavior

### ✅ Server Features
- Health check endpoint
- Info endpoint with method listing
- Proper FastMCP integration
- Server configuration options

### ✅ Edge Cases
- Division by zero error handling
- Empty parameter lists
- Complex return types
- Async method execution

## Example Usage

Here's how to use AutoMCP with your own classes:

```python
from auto_mcp import AutoMCP

# Your class
class MyService:
    def process_data(self, data: dict) -> dict:
        """Process some data."""
        return {"processed": True, "data": data}
    
    def get_status(self) -> str:
        """Get service status."""
        return "running"

# Create AutoMCP server
service = MyService()
auto_mcp = AutoMCP(
    instance=service,
    server_name="My Service API",
    exclude_methods=["private_method"]  # Optional
)

# Start the server
auto_mcp.run(host="localhost", port=8000)
```

## Testing Your Own Classes

To test your own classes with AutoMCP:

1. **Create a test instance of your class**
2. **Add it to the test suite** by modifying `test_auto_mcp.py`
3. **Run the tests** to verify proper registration
4. **Start a test server** to test actual HTTP endpoints

Example:
```python
# Add to test_auto_mcp.py
def test_my_service():
    """Test MyService class."""
    service = MyService()
    auto_mcp = AutoMCP(service, server_name="Test My Service")
    
    methods = auto_mcp.list_registered_methods()
    print(f"MyService registered {len(methods)} methods:")
    for method in methods:
        print(f"  - {method['name']}: {method['description']}")
```

## Requirements

The test scripts require:
- Python 3.7+
- `fastmcp` package
- `uvicorn` package
- `requests` package (for client tests)

Install with:
```bash
pip install fastmcp uvicorn requests
```

## Troubleshooting

### Server Won't Start
- Check if the port is already in use
- Verify all dependencies are installed
- Look for error messages in the console output

### Tests Fail
- Make sure you're in the correct directory
- Verify the `auto_mcp.py` file is present and working
- Check Python version compatibility

### Client Tests Fail
- Ensure a server is running before running client tests
- Verify the server URL is correct
- Check firewall/network settings

## Advanced Testing

### Custom Test Classes
You can create your own test classes to verify specific functionality:

```python
class MyTestClass:
    def my_method(self, param: str) -> dict:
        """My test method."""
        return {"result": param}

# Test it
test_instance = MyTestClass()
auto_mcp = AutoMCP(test_instance, server_name="My Test")
# Verify registration, test methods, etc.
```

### Performance Testing
For performance testing, you can:
1. Create classes with many methods
2. Test registration time
3. Test method call performance
4. Monitor memory usage

### Integration Testing
Test integration with your existing systems:
1. Use real data classes
2. Test with actual use cases
3. Verify error handling with real scenarios
4. Test concurrent access if needed

## Contributing

When adding new tests:
1. Follow the existing naming conventions
2. Add comprehensive docstrings
3. Test both success and failure cases
4. Update this README if adding new features
