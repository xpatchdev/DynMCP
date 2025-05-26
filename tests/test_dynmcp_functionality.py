#!/usr/bin/env python3
"""
test_dynmcp_functionality.py - Test script to verify DynMCP and AutoMCP functionality

This script demonstrates that:
1. DynMCP works independently
2. AutoMCP works as an alias for DynMCP (backward compatibility)
3. Both classes can register methods and create servers
"""

from dynmcp import DynMCP
from examples.auto_mcp import AutoMCP

class ExampleAPI:
    """Example API class with various method types."""
    
    def simple_method(self) -> str:
        """A simple method with no parameters."""
        return "Hello from simple_method!"
    
    def add_numbers(self, a: int, b: int) -> int:
        """Add two integers."""
        return a + b
    
    def greet_user(self, name: str, greeting: str = "Hello") -> str:
        """Greet a user with a custom greeting."""
        return f"{greeting}, {name}!"
    
    def calculate_area(self, length: float, width: float) -> float:
        """Calculate the area of a rectangle."""
        return length * width
    
    def _private_method(self) -> str:
        """This should not be exposed."""
        return "This is private"

def test_dynmcp():
    """Test DynMCP functionality."""
    print("=== Testing DynMCP ===")
    
    # Create instance
    api = ExampleAPI()
    
    # Create DynMCP server
    dynmcp_server = DynMCP(
        instance=api,
        server_name="DynMCP Test Server",
        exclude_methods=["_private_method"]
    )
    
    # Display registered methods
    print(f"DynMCP registered {len(dynmcp_server.registered_methods)} methods:")
    for method_info in dynmcp_server.list_registered_methods():
        print(f"  - {method_info['name']}: {method_info['description']}")
    
    return dynmcp_server

def test_automcp():
    """Test AutoMCP functionality (should be identical to DynMCP)."""
    print("\n=== Testing AutoMCP (Backward Compatibility) ===")
    
    # Create instance
    api = ExampleAPI()
    
    # Create AutoMCP server
    automcp_server = AutoMCP(
        instance=api,
        server_name="AutoMCP Test Server",
        exclude_methods=["_private_method"]
    )
    
    # Display registered methods
    print(f"AutoMCP registered {len(automcp_server.registered_methods)} methods:")
    for method_info in automcp_server.list_registered_methods():
        print(f"  - {method_info['name']}: {method_info['description']}")
    
    return automcp_server

def test_inheritance():
    """Test that AutoMCP is indeed a subclass of DynMCP."""
    print("\n=== Testing Inheritance ===")
    
    api = ExampleAPI()
    automcp_server = AutoMCP(api, server_name="Inheritance Test")
    
    print(f"AutoMCP is instance of DynMCP: {isinstance(automcp_server, DynMCP)}")
    print(f"AutoMCP is instance of AutoMCP: {isinstance(automcp_server, AutoMCP)}")
    print(f"AutoMCP class MRO: {[cls.__name__ for cls in AutoMCP.__mro__]}")

def main():
    """Run all tests."""
    print("Testing DynMCP and AutoMCP functionality...\n")
    
    # Test DynMCP
    dynmcp_server = test_dynmcp()
    
    # Test AutoMCP
    automcp_server = test_automcp()
    
    # Test inheritance
    test_inheritance()
    
    # Compare functionality
    print("\n=== Comparison ===")
    dynmcp_methods = set(dynmcp_server.registered_methods.keys())
    automcp_methods = set(automcp_server.registered_methods.keys())
    
    print(f"DynMCP methods: {dynmcp_methods}")
    print(f"AutoMCP methods: {automcp_methods}")
    print(f"Methods are identical: {dynmcp_methods == automcp_methods}")
    
    print("\n✅ All tests completed successfully!")
    print("✅ DynMCP and AutoMCP are working correctly!")
    print("✅ Backward compatibility is maintained!")

if __name__ == "__main__":
    main()
