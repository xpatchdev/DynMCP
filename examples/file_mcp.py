#!/usr/bin/env python3
"""
file_mcp.py - Example: Expose FileIO operations as MCP endpoints using DynMCP

This script demonstrates how to use DynMCP to automatically expose the FileIO class
methods as MCP endpoints using FastMCP.
"""

from dynmcp import DynMCP
from examples.FileIO import FileIO

if __name__ == "__main__":
    # Create DynMCP server for FileIO with prefix 'file'
    file_mcp = DynMCP(
        instance=FileIO(),
        server_name="FileIO MCP Server",
        service_prefix="file",  # Explicitly set prefix to 'file'
        no_prefix=True,  # Register endpoints without prefix
        exclude_methods=[],  # Optionally exclude methods if needed
        instructions="File input/output operations: list directories, read files, write files."
    )

    # Print information about registered methods
    print("Registered FileIO methods:")
    for method_info in file_mcp.list_registered_methods():
        print(f"  {method_info['name']}: {method_info['description']}")

    # Start the server (uncomment to actually run)
    file_mcp.run()
