#!/usr/bin/env python3
"""
dynmcp.py - DynMCP Class for dynamically exposing class methods as MCP endpoints

This module provides the DynMCP class which takes any class instance and 
automatically exposes all its public methods as MCP endpoints using FastMCP's 
@mcp.tool decorator.
"""

import inspect
import json
import functools
import os
from typing import Any, Dict, List, Optional, Union, get_type_hints
from mcp.server.fastmcp import FastMCP


class DynMCP:
    """
    DynMCP dynamically exposes all public methods of a class instance as MCP endpoints.
    
    This class uses reflection to discover public methods of the given instance and 
    dynamically creates MCP tool endpoints for each method using FastMCP.
    
    Example:
        # Create a class with methods you want to expose
        class Calculator:
            def add(self, a: int, b: int) -> int:
                '''Add two numbers'''
                return a + b
                
            def multiply(self, a: float, b: float) -> float:
                '''Multiply two numbers'''
                return a * b
        
        # Create DynMCP instance
        calc = Calculator()
        dyn_mcp = DynMCP(calc, server_name="Calculator Server")
        
        # Start the server
        dyn_mcp.run()
    """
    
    def __init__(self, 
                 instance: Any, 
                 server_name: str = "DynMCP Server",
                 include_private: bool = False,
                 method_filter: Optional[List[str]] = None,
                 exclude_methods: Optional[List[str]] = None,
                 fastmcp_port: int = 8000,
                 ip_addr: str = "localhost",
                 instructions: Optional[str] = None,
                 service_prefix: Optional[str] = None,
                 no_prefix: bool = False):
        """
        Initialize DynMCP with a class instance.
        
        Args:
            instance: The class instance whose methods should be exposed
            server_name: Name for the FastMCP server
            include_private: Whether to include methods starting with underscore
            method_filter: List of specific method names to include (None = all public)
            exclude_methods: List of method names to exclude
            fastmcp_port: Port to bind the FastMCP server to
            ip_addr: IP address to bind the FastMCP server to
            instructions: Instructions to be passed to the FastMCP constructor
            service_prefix: Prefix for all endpoint names (default: first 3 letters of server_name, lowercased)
            no_prefix: If True, do not use any prefix for endpoint names (just the function name)
        """
        self.instance = instance
        self.server_name = server_name
        self.include_private = include_private
        self.method_filter = method_filter or []
        self.exclude_methods = exclude_methods or []
        self.fastmcp_port = fastmcp_port
        self.ip_addr = ip_addr
        self.instructions = instructions
        self.no_prefix = no_prefix
        if service_prefix is not None and service_prefix.strip():
            self.service_prefix = service_prefix.strip().lower()
        else:
            self.service_prefix = server_name[:3].lower() if server_name else "svc"
        
        # Initialize FastMCP server with instructions if provided
        if instructions is not None:
            self.mcp = FastMCP(server_name, instructions=instructions)
        else:
            self.mcp = FastMCP(server_name)
        
        # Store registered methods for reference
        self.registered_methods: Dict[str, Any] = {}
        
        # Automatically register all eligible methods
        self._register_methods()
    
    def _get_eligible_methods(self) -> List[tuple[str, Any]]:
        """
        Get all eligible methods from the instance.
        
        Returns:
            List of (method_name, method_object) tuples
        """
        methods = []
        
        # Get all attributes of the instance
        for name in dir(self.instance):
            # Skip magic methods
            if name.startswith('__') and name.endswith('__'):
                continue
                
            # Skip private methods unless explicitly included
            if name.startswith('_') and not self.include_private:
                continue
                
            # Skip if in exclude list
            if name in self.exclude_methods:
                continue
                
            # If method_filter is provided, only include those methods
            if self.method_filter and name not in self.method_filter:
                continue
                
            attr = getattr(self.instance, name)
            
            # Only include callable methods
            if callable(attr) and not isinstance(attr, type):
                methods.append((name, attr))
                
        return methods
    
    def _get_method_signature_info(self, method: Any) -> Dict[str, Any]:
        """
        Extract signature information from a method for MCP tool registration.
        
        Args:
            method: The method to analyze
            
        Returns:
            Dictionary with signature information
        """
        sig = inspect.signature(method)
        type_hints = get_type_hints(method)
        
        # Extract parameters (skip 'self' for instance methods)
        params = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_info = {
                'name': param_name,
                'annotation': type_hints.get(param_name, Any),
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'required': param.default == inspect.Parameter.empty
            }
            params[param_name] = param_info
            
        return {
            'parameters': params,
            'return_type': type_hints.get('return', Any),
            'docstring': inspect.getdoc(method) or f"Auto-generated endpoint for {method.__name__}"
        }
    
    def _create_mcp_wrapper(self, method_name: str, method: Any) -> Any:
        """
        Create a wrapper function for the method that can be used with @mcp.tool.
        
        Args:
            method_name: Name of the method
            method: The actual method object
            
        Returns:
            Wrapper function suitable for MCP tool registration
        """
        sig_info = self._get_method_signature_info(method)
        
        # Create wrapper function with proper signature
        @functools.wraps(method)
        async def wrapper(*args, **kwargs):
            try:
                # Call the original method
                result = method(*args, **kwargs)
                
                # If the result is a coroutine, await it
                if inspect.iscoroutine(result):
                    result = await result
                
                # Ensure we return a JSON-serializable result
                if isinstance(result, (dict, list, str, int, float, bool)) or result is None:
                    return result
                else:
                    # Try to convert to string if not JSON serializable
                    return str(result)
                    
            except Exception as e:
                # Return error information in a structured format
                return {
                    "error": True,
                    "message": str(e),
                    "method": method_name,
                    "type": type(e).__name__
                }
        
        # Set proper function name and docstring
        wrapper.__name__ = method_name
        wrapper.__doc__ = sig_info['docstring']
        
        return wrapper
    
    def _register_methods(self):
        """Register all eligible methods as MCP tools."""
        eligible_methods = self._get_eligible_methods()
        
        print(f"DynMCP: Registering {len(eligible_methods)} methods as MCP endpoints...")
        
        for method_name, method in eligible_methods:
            try:
                # Create wrapper function
                wrapper = self._create_mcp_wrapper(method_name, method)
                
                # Get signature info for the tool decorator
                sig_info = self._get_method_signature_info(method)
                
                # Compose endpoint name with or without prefix
                if self.no_prefix:
                    endpoint_name = method_name
                else:
                    endpoint_name = f"{self.service_prefix}_{method_name}"
                
                # Register as MCP tool
                tool_decorator = self.mcp.tool(
                    name=endpoint_name,
                    description=sig_info['docstring']
                )
                
                # Apply decorator and store
                decorated_wrapper = tool_decorator(wrapper)
                self.registered_methods[endpoint_name] = {
                    'original_method': method,
                    'wrapper': wrapper,
                    'decorated_wrapper': decorated_wrapper,
                    'signature_info': sig_info
                }
                
                print(f"  ✓ Registered: {endpoint_name}")
                
            except Exception as e:
                print(f"  ✗ Failed to register {method_name}: {str(e)}")
    
    def list_registered_methods(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered methods.
        
        Returns:
            List of dictionaries with method information
        """
        methods_info = []
        for name, info in self.registered_methods.items():
            methods_info.append({
                'name': name,
                'description': info['signature_info']['docstring'],
                'parameters': list(info['signature_info']['parameters'].keys()),
                'return_type': str(info['signature_info']['return_type'])
            })
        return methods_info
    
    def add_health_check(self):
        """Add a health check endpoint to the MCP server."""
        @self.mcp.tool(name="health", description="Health check for the DynMCP server")
        async def health_check() -> Dict[str, Any]:
            """Health check endpoint."""
            return {
                "status": "ok",
                "server": self.server_name,
                "registered_methods": len(self.registered_methods),
                "methods": list(self.registered_methods.keys())
            }
    
    def add_info_endpoint(self):
        """Add an info endpoint that lists all registered methods."""
        @self.mcp.tool(name="info", description="Get information about registered methods")
        async def info() -> Dict[str, Any]:
            """Get information about all registered methods."""
            return {
                "server_name": self.server_name,
                "instance_type": type(self.instance).__name__,
                "total_methods": len(self.registered_methods),
                "methods": self.list_registered_methods()
            }
    
    def run(self, add_health_check: bool = True, add_info: bool = True):
        """
        Start the FastMCP server.
        
        Args:
            add_health_check: Whether to add a health check endpoint
            add_info: Whether to add an info endpoint
        """
        if add_health_check:
            self.add_health_check()
        if add_info:
            self.add_info_endpoint()
        
        # Set environment variables for FastMCP host and port
        os.environ["FASTMCP_HOST"] = str(self.ip_addr)
        os.environ["FASTMCP_PORT"] = str(self.fastmcp_port)
        print(f"\n=== Starting {self.server_name} ===")
        print(f"IP Address: {self.ip_addr}")
        print(f"Port: {self.fastmcp_port}")
        print(f"Registered Methods: {len(self.registered_methods)}")
        print(f"Instance Type: {type(self.instance).__name__}")
        print("\nAvailable Endpoints:")
        for method_name, info in self.registered_methods.items():
            params = list(info['signature_info']['parameters'].keys())
            print(f"  - {method_name}({', '.join(params)})")
        print(f"\nServer is running at http://{self.ip_addr}:{self.fastmcp_port}")
        print("Press Ctrl+C to stop.")
        self.mcp.run(transport="sse")






