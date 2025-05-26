#!/usr/bin/env python3
"""
test_auto_mcp_client.py - Client test script for AutoMCP server

This script tests an AutoMCP server by making HTTP requests to its endpoints.
Run this while an AutoMCP server is running to test the actual MCP functionality.
"""

import json
import requests
import time
import sys
from typing import Dict, Any, Optional

class AutoMCPClient:
    """Simple client for testing AutoMCP servers."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client with the server URL."""
        self.base_url = base_url.rstrip('/')
        
    def check_server_health(self) -> bool:
        """Check if the server is responsive."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def call_method(self, method_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Call a method on the AutoMCP server."""
        try:
            url = f"{self.base_url}/{method_name}"
            
            # Send as JSON POST request
            response = requests.post(
                url,
                json=kwargs,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": True,
                    "status_code": response.status_code,
                    "message": response.text
                }
                
        except Exception as e:
            return {
                "error": True,
                "message": str(e),
                "type": "client_error"
            }
    
    def get_server_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the server and its methods."""
        return self.call_method("info")
    
    def test_calculator_methods(self):
        """Test calculator methods if available."""
        print("\n=== Testing Calculator Methods ===")
        
        # Test addition
        result = self.call_method("add", a=5, b=3)
        if result and not result.get("error"):
            if result == 8:
                print("  ✓ add(5, 3) = 8")
            else:
                print(f"  ✗ add(5, 3) returned {result}, expected 8")
        else:
            print(f"  ✗ add failed: {result}")
        
        # Test subtraction
        result = self.call_method("subtract", a=10, b=4)
        if result and not result.get("error"):
            if result == 6:
                print("  ✓ subtract(10, 4) = 6")
            else:
                print(f"  ✗ subtract(10, 4) returned {result}, expected 6")
        else:
            print(f"  ✗ subtract failed: {result}")
        
        # Test multiplication
        result = self.call_method("multiply", a=3.5, b=2.0)
        if result and not result.get("error"):
            if abs(result - 7.0) < 0.001:
                print("  ✓ multiply(3.5, 2.0) = 7.0")
            else:
                print(f"  ✗ multiply(3.5, 2.0) returned {result}, expected 7.0")
        else:
            print(f"  ✗ multiply failed: {result}")
        
        # Test division
        result = self.call_method("divide", a=15.0, b=3.0)
        if result and not result.get("error"):
            if abs(result - 5.0) < 0.001:
                print("  ✓ divide(15.0, 3.0) = 5.0")
            else:
                print(f"  ✗ divide(15.0, 3.0) returned {result}, expected 5.0")
        else:
            print(f"  ✗ divide failed: {result}")
        
        # Test error handling - division by zero
        result = self.call_method("divide", a=10.0, b=0.0)
        if result and result.get("error"):
            print(f"  ✓ divide by zero error handled: {result.get('message')}")
        else:
            print(f"  ✗ divide by zero should have returned error, got: {result}")
    
    def test_string_methods(self):
        """Test string processing methods if available."""
        print("\n=== Testing String Methods ===")
        
        # Test uppercase
        result = self.call_method("uppercase", text="hello world")
        if result and not result.get("error"):
            if result == "HELLO WORLD":
                print("  ✓ uppercase('hello world') = 'HELLO WORLD'")
            else:
                print(f"  ✗ uppercase returned {result}")
        else:
            print(f"  ✗ uppercase failed: {result}")
        
        # Test lowercase
        result = self.call_method("lowercase", text="HELLO WORLD")
        if result and not result.get("error"):
            if result == "hello world":
                print("  ✓ lowercase('HELLO WORLD') = 'hello world'")
            else:
                print(f"  ✗ lowercase returned {result}")
        else:
            print(f"  ✗ lowercase failed: {result}")
        
        # Test count_words
        result = self.call_method("count_words", text="The quick brown fox")
        if result and not result.get("error"):
            if result == 4:
                print("  ✓ count_words('The quick brown fox') = 4")
            else:
                print(f"  ✗ count_words returned {result}, expected 4")
        else:
            print(f"  ✗ count_words failed: {result}")


def main():
    """Main test function."""
    print("AutoMCP Client Test Script")
    print("=" * 50)
    
    # Get server URL from command line or use default
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    client = AutoMCPClient(server_url)
    
    print(f"Testing server at: {server_url}")
    
    # Check if server is running
    print("\nChecking server health...")
    if not client.check_server_health():
        print("✗ Server is not responding!")
        print("Make sure you have an AutoMCP server running.")
        print("You can start one with:")
        print("  python3 test_auto_mcp.py --run-server")
        sys.exit(1)
    
    print("✓ Server is responding")
    
    # Get server information
    print("\nGetting server information...")
    info = client.get_server_info()
    if info and not info.get("error"):
        print(f"  Server: {info.get('server_name', 'Unknown')}")
        print(f"  Instance Type: {info.get('instance_type', 'Unknown')}")
        print(f"  Total Methods: {info.get('total_methods', 0)}")
        
        methods = info.get('methods', [])
        if methods:
            print("  Available Methods:")
            for method in methods:
                params = ', '.join(method.get('parameters', []))
                print(f"    - {method['name']}({params})")
    else:
        print(f"  ✗ Failed to get server info: {info}")
    
    # Test health endpoint
    print("\nTesting health endpoint...")
    health = client.call_method("health")
    if health and not health.get("error"):
        print(f"  ✓ Health check: {health.get('status')}")
        print(f"  ✓ Registered methods: {health.get('registered_methods')}")
    else:
        print(f"  ✗ Health check failed: {health}")
    
    # Test calculator methods (if available)
    client.test_calculator_methods()
    
    # Test string methods (if available)
    client.test_string_methods()
    
    print("\n" + "=" * 50)
    print("✓ Client testing completed!")


if __name__ == "__main__":
    main()
