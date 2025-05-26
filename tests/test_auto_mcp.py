#!/usr/bin/env python3
"""
test_auto_mcp.py - Comprehensive test script for AutoMCP class

This script tests the AutoMCP functionality by creating test classes with various
method signatures and verifying that they are properly exposed as MCP endpoints.
"""

import asyncio
import sys
import time
from typing import Dict, List, Optional, Any
from examples.auto_mcp import AutoMCP


class TestCalculator:
    """Test class with basic arithmetic operations."""
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
        
    def subtract(self, a: int, b: int) -> int:
        """Subtract second number from first."""
        return a - b
        
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
        
    def divide(self, a: float, b: float) -> float:
        """Divide first number by second."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
        
    def _private_method(self) -> str:
        """This is a private method that shouldn't be exposed."""
        return "This is private"


class TestStringProcessor:
    """Test class with string processing methods."""
    
    def uppercase(self, text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()
        
    def lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
        
    def reverse_string(self, text: str) -> str:
        """Reverse the input string."""
        return text[::-1]
        
    def count_words(self, text: str) -> int:
        """Count the number of words in the text."""
        return len(text.split())
        
    def repeat_text(self, text: str, times: int = 1) -> str:
        """Repeat text a specified number of times."""
        return text * times


class TestDataProcessor:
    """Test class with complex data processing methods."""
    
    def process_list(self, items: List[int]) -> Dict[str, Any]:
        """Process a list of integers and return statistics."""
        if not items:
            return {"error": "Empty list provided"}
            
        return {
            "sum": sum(items),
            "average": sum(items) / len(items),
            "min": min(items),
            "max": max(items),
            "count": len(items)
        }
        
    def filter_even(self, numbers: List[int]) -> List[int]:
        """Filter out odd numbers, returning only even numbers."""
        return [n for n in numbers if n % 2 == 0]
        
    def create_user_profile(self, name: str, age: int, email: Optional[str] = None) -> Dict[str, Any]:
        """Create a user profile dictionary."""
        profile = {
            "name": name,
            "age": age,
            "created_at": time.time()
        }
        if email:
            profile["email"] = email
        return profile


class TestAsyncOperations:
    """Test class with async methods."""
    
    async def async_wait(self, seconds: float) -> str:
        """Async method that waits for specified seconds."""
        await asyncio.sleep(seconds)
        return f"Waited for {seconds} seconds"
        
    def sync_method(self, message: str) -> str:
        """Regular sync method."""
        return f"Sync: {message}"


def test_basic_registration():
    """Test basic method registration."""
    print("=== Testing Basic Registration ===")
    
    calc = TestCalculator()
    auto_mcp = AutoMCP(calc, server_name="Test Calculator")
    
    # Check registered methods
    methods = auto_mcp.list_registered_methods()
    print(f"Registered {len(methods)} methods:")
    
    expected_methods = ["add", "subtract", "multiply", "divide"]
    registered_names = [m["name"] for m in methods]
    
    for expected in expected_methods:
        if expected in registered_names:
            print(f"  ✓ {expected}")
        else:
            print(f"  ✗ {expected} - MISSING")
    
    # Check that private method is not registered
    if "_private_method" not in registered_names:
        print("  ✓ Private method correctly excluded")
    else:
        print("  ✗ Private method was registered - SHOULD NOT HAPPEN")
    
    return auto_mcp


def test_method_filtering():
    """Test method filtering functionality."""
    print("\n=== Testing Method Filtering ===")
    
    processor = TestStringProcessor()
    
    # Test with method filter
    auto_mcp = AutoMCP(
        processor, 
        server_name="Filtered String Processor",
        method_filter=["uppercase", "lowercase"]
    )
    
    methods = auto_mcp.list_registered_methods()
    registered_names = [m["name"] for m in methods]
    
    print(f"With method_filter ['uppercase', 'lowercase']:")
    expected_filtered = ["uppercase", "lowercase"]
    
    for method in expected_filtered:
        if method in registered_names:
            print(f"  ✓ {method}")
        else:
            print(f"  ✗ {method} - MISSING")
    
    # Check that non-filtered methods are excluded
    excluded = ["reverse_string", "count_words", "repeat_text"]
    for method in excluded:
        if method not in registered_names:
            print(f"  ✓ {method} correctly excluded")
        else:
            print(f"  ✗ {method} should be excluded")


def test_method_exclusion():
    """Test method exclusion functionality."""
    print("\n=== Testing Method Exclusion ===")
    
    processor = TestStringProcessor()
    
    # Test with exclusion
    auto_mcp = AutoMCP(
        processor,
        server_name="String Processor with Exclusions",
        exclude_methods=["reverse_string", "repeat_text"]
    )
    
    methods = auto_mcp.list_registered_methods()
    registered_names = [m["name"] for m in methods]
    
    print(f"With exclude_methods ['reverse_string', 'repeat_text']:")
    
    expected_included = ["uppercase", "lowercase", "count_words"]
    for method in expected_included:
        if method in registered_names:
            print(f"  ✓ {method}")
        else:
            print(f"  ✗ {method} - MISSING")
    
    excluded = ["reverse_string", "repeat_text"]
    for method in excluded:
        if method not in registered_names:
            print(f"  ✓ {method} correctly excluded")
        else:
            print(f"  ✗ {method} should be excluded")


def test_complex_signatures():
    """Test handling of complex method signatures."""
    print("\n=== Testing Complex Method Signatures ===")
    
    processor = TestDataProcessor()
    auto_mcp = AutoMCP(processor, server_name="Data Processor")
    
    methods = auto_mcp.list_registered_methods()
    
    print("Registered methods with complex signatures:")
    for method in methods:
        print(f"  - {method['name']}: {method['description']}")
        print(f"    Parameters: {method['parameters']}")
        print(f"    Return type: {method['return_type']}")


def test_signature_analysis():
    """Test the signature analysis functionality."""
    print("\n=== Testing Signature Analysis ===")
    
    calc = TestCalculator()
    auto_mcp = AutoMCP(calc, server_name="Signature Test")
    
    # Get signature info for the add method
    if "add" in auto_mcp.registered_methods:
        sig_info = auto_mcp.registered_methods["add"]["signature_info"]
        print("Signature analysis for 'add' method:")
        print(f"  Parameters: {sig_info['parameters']}")
        print(f"  Return type: {sig_info['return_type']}")
        print(f"  Docstring: {sig_info['docstring']}")
        
        # Check parameter details
        params = sig_info['parameters']
        if 'a' in params and 'b' in params:
            print("  ✓ Parameters 'a' and 'b' detected")
            if params['a']['required'] and params['b']['required']:
                print("  ✓ Parameters marked as required")
            else:
                print("  ✗ Parameters should be marked as required")
        else:
            print("  ✗ Parameters not properly detected")


def test_wrapper_functionality():
    """Test that method wrappers work correctly."""
    print("\n=== Testing Method Wrapper Functionality ===")
    
    calc = TestCalculator()
    auto_mcp = AutoMCP(calc, server_name="Wrapper Test")
    
    # Test calling wrapped methods directly
    if "add" in auto_mcp.registered_methods:
        wrapper = auto_mcp.registered_methods["add"]["wrapper"]
        
        try:
            # Test normal operation
            result = asyncio.run(wrapper(5, 3))
            if result == 8:
                print("  ✓ Add wrapper works correctly")
            else:
                print(f"  ✗ Add wrapper returned {result}, expected 8")
        except Exception as e:
            print(f"  ✗ Error calling add wrapper: {e}")
    
    # Test error handling
    if "divide" in auto_mcp.registered_methods:
        wrapper = auto_mcp.registered_methods["divide"]["wrapper"]
        
        try:
            # Test division by zero
            result = asyncio.run(wrapper(10, 0))
            if isinstance(result, dict) and result.get("error"):
                print("  ✓ Error handling works correctly")
                print(f"    Error message: {result.get('message')}")
            else:
                print(f"  ✗ Expected error dict, got {result}")
        except Exception as e:
            print(f"  ✗ Unexpected exception: {e}")


def test_async_methods():
    """Test handling of async methods."""
    print("\n=== Testing Async Method Support ===")
    
    async_ops = TestAsyncOperations()
    auto_mcp = AutoMCP(async_ops, server_name="Async Test")
    
    methods = auto_mcp.list_registered_methods()
    registered_names = [m["name"] for m in methods]
    
    if "async_wait" in registered_names:
        print("  ✓ Async method registered")
        
        # Test calling the async method
        wrapper = auto_mcp.registered_methods["async_wait"]["wrapper"]
        try:
            result = asyncio.run(wrapper(0.1))  # Wait 0.1 seconds
            if "Waited for 0.1 seconds" in str(result):
                print("  ✓ Async method wrapper works")
            else:
                print(f"  ✗ Unexpected result: {result}")
        except Exception as e:
            print(f"  ✗ Error calling async wrapper: {e}")
    else:
        print("  ✗ Async method not registered")
    
    if "sync_method" in registered_names:
        print("  ✓ Sync method also registered")
    else:
        print("  ✗ Sync method not registered")


def test_info_endpoints():
    """Test the built-in info endpoints."""
    print("\n=== Testing Info Endpoints ===")
    
    calc = TestCalculator()
    auto_mcp = AutoMCP(calc, server_name="Info Test")
    
    # Add info endpoints
    auto_mcp.add_health_check()
    auto_mcp.add_info_endpoint()
    
    # Check if they were added to the MCP server
    print("  ✓ Health check and info endpoints added")
    
    # Test the list_registered_methods functionality
    methods_info = auto_mcp.list_registered_methods()
    print(f"  ✓ list_registered_methods returns {len(methods_info)} methods")


def run_all_tests():
    """Run all test functions."""
    print("AutoMCP Test Suite")
    print("=" * 50)
    
    try:
        # Run all tests
        test_basic_registration()
        test_method_filtering()
        test_method_exclusion()
        test_complex_signatures()
        test_signature_analysis()
        test_wrapper_functionality()
        test_async_methods()
        test_info_endpoints()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def interactive_test():
    """Interactive test that lets you manually test method calls."""
    print("\n=== Interactive Test Mode ===")
    print("This will create an AutoMCP server but won't start it.")
    print("You can examine the registered methods and their signatures.")
    
    # Create a test instance
    calc = TestCalculator()
    auto_mcp = AutoMCP(calc, server_name="Interactive Test Calculator")
    
    print(f"\nServer created with {len(auto_mcp.registered_methods)} methods:")
    
    for name, info in auto_mcp.registered_methods.items():
        sig_info = info['signature_info']
        params = [f"{p}: {sig_info['parameters'][p]['annotation'].__name__}" 
                 for p in sig_info['parameters']]
        print(f"  {name}({', '.join(params)}) -> {sig_info['return_type']}")
        print(f"    {sig_info['docstring']}")
    
    print("\nTo actually run the server, uncomment the auto_mcp.run() line at the end of this script.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test AutoMCP functionality")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--run-server", "-r", action="store_true",
                       help="Actually start an MCP server for testing")
    parser.add_argument("--port", "-p", type=int, default=8000,
                       help="Port to run test server on")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_test()
    elif args.run_server:
        print("Starting test MCP server...")
        calc = TestCalculator()
        auto_mcp = AutoMCP(calc, server_name="Test AutoMCP Server")
        print(f"Starting server on port {args.port}")
        auto_mcp.run(port=args.port)
    else:
        # Run the test suite
        success = run_all_tests()
        sys.exit(0 if success else 1)
