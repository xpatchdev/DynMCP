#!/usr/bin/env python3
"""
auto_mcp.py - AutoMCP Server using DynMCP for automatically exposing class methods as MCP endpoints

This module demonstrates the usage of the DynMCP class to automatically expose 
class methods as MCP endpoints using FastMCP.
"""

from dynmcp import DynMCP

# CalculatorMCP is now an alias for DynMCP for backward compatibility
class CalculatorMCP(DynMCP):
    """
    CalculatorMCP automatically exposes all public methods of a class instance as MCP endpoints.
    
    This class uses reflection to discover public methods of the given instance and 
    dynamically creates MCP tool endpoints for each method using FastMCP.
    
    Note: CalculatorMCP is now an alias for DynMCP for backward compatibility.
    
    Example:
        # Create a class with methods you want to expose
        class Calculator:
            def add(self, a: int, b: int) -> int:
                '''Add two numbers'''
                return a + b
                
            def multiply(self, a: float, b: float) -> float:
                '''Multiply two numbers'''
                return a * b
        
        # Create CalculatorMCP instance
        calc = Calculator()
        auto_mcp = CalculatorMCP(calc, server_name="Calculator Server")
        
        # Start the server
        auto_mcp.run()
    """
    pass


# Example usage and test classes
if __name__ == "__main__":
    # Example class to demonstrate CalculatorMCP
    class Calculator:
        """A simple calculator class for demonstration."""
        
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
            
        def power(self, base: float, exponent: float) -> float:
            """Raise base to the power of exponent."""
            return base ** exponent
            
        def _private_method(self) -> str:
            """This is a private method that shouldn't be exposed."""
            return "This is private"
    
    # Create calculator instance
    calc = Calculator()
    
    # Create AutoMCP server
    auto_mcp = CalculatorMCP(
        instance=calc,
        server_name="Calculator MCP Server",
        exclude_methods=["_private_method"]  # Explicitly exclude private methods
    )
    
    # Print information about registered methods
    print("Registered methods:")
    for method_info in auto_mcp.list_registered_methods():
        print(f"  {method_info['name']}: {method_info['description']}")
    
    # Start the server (uncomment to actually run)
    # auto_mcp.run(port=8001)