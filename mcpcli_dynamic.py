#!/usr/bin/env python3
"""
MCP CLI Dynamic - A command-line interface for MCP tools.
Dynamically discovers and runs tools from an MCP server.

Features:
- If no arguments given or just a server URL, shows all available tools and example usage
- If just the tool name is supplied, runs the tool if no arguments required
- If tool name is supplied but required arguments are missing, shows usage
- If tool name and required arguments are supplied, runs the tool
- Supports different transport protocols (HTTP, SSE, WebSocket)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from mcp_tool_discovery_client import MCPToolDiscoveryClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcpcli-dynamic")

class MCPCliDynamic:
    def __init__(self, timeout: int = 5, transport: str = "sse"):
        # Get the MCP server URL from the environment variable or use the default
        url = os.environ.get("MCP_HOST", "http://localhost:8000")
        self.client = MCPToolDiscoveryClient(url, timeout, transport)
        self.tools = None

    async def discover_tools(self) -> None:
        """Discover available tools from the MCP server."""
        try:
            self.tools = await self.client.discover_tools()
            logger.debug(f"Discovered {len(self.tools)} tools")
        except asyncio.TimeoutError as e:
            logger.error(f"Connection timeout: {str(e)}")
            raise
        except ConnectionError as e:
            # Log and re-raise connection errors 
            logger.error(f"Connection error: {str(e)}")
            raise
        except Exception as e:
            # Try to extract the actual error message if it's a nested exception
            if hasattr(e, '__context__') and e.__context__ is not None:
                actual_error = e.__context__
                logger.error(f"Failed to discover tools: {actual_error}")
                raise
            else:
                logger.error(f"Failed to discover tools: {e}")
                raise

    def get_example_usage(self, tool) -> str:
        """Generate example usage for a tool based on its schema."""
        example = f"  {sys.argv[0]} {tool.name}"
        
        if hasattr(tool, 'inputSchema') and isinstance(tool.inputSchema, dict):
            schema = tool.inputSchema
            if 'properties' in schema:
                for param_name, param_info in schema['properties'].items():
                    is_required = 'required' in schema and param_name in schema['required']
                    param_type = param_info.get('type', 'string')
                    
                    example_value = ""
                    if param_type == 'string':
                        example_value = '"example_value"'
                    elif param_type == 'integer':
                        example_value = "123"
                    elif param_type == 'number':
                        example_value = "3.14"
                    elif param_type == 'boolean':
                        example_value = "true"
                    elif param_type == 'array':
                        example_value = '"item1" "item2"'
                    elif param_type == 'object':
                        example_value = '\'{"key":"value"}\''
                    
                    if is_required:
                        example += f" --{param_name} {example_value}"
                    else:
                        example += f" [--{param_name} {example_value}]"
        
        return example

    def display_all_tools(self) -> None:
        """Display all available tools with descriptions and example usage."""
        if not self.tools:
            print("No tools available.")
            return
        
        print(f"Available tools ({len(self.tools)}):")
        print("=" * 50)
        
        for tool in self.tools:
            print(f"\n{tool.name}: {tool.description}")
            
            if hasattr(tool, 'inputSchema') and isinstance(tool.inputSchema, dict):
                schema = tool.inputSchema
                
                if 'properties' in schema:
                    print("  Parameters:")
                    for param_name, param_info in schema['properties'].items():
                        param_type = param_info.get('type', 'string')
                        is_required = 'required' in schema and param_name in schema['required']
                        
                        print(f"    --{param_name} ({param_type})", end="")
                        if is_required:
                            print(" (required)", end="")
                        print()
                        
                        if 'description' in param_info:
                            print(f"      {param_info['description']}")
                
                if 'required' in schema and schema['required']:
                    print(f"  Required parameters: {', '.join(schema['required'])}")
            
            print("\n  Example usage:")
            print(f"    {self.get_example_usage(tool)}")
            print("-" * 50)

    def display_tool_usage(self, tool_name: str) -> None:
        """Display usage information for a specific tool."""
        if not self.tools:
            print(f"Tool '{tool_name}' not found.")
            return
        
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            print(f"Tool '{tool_name}' not found.")
            return
        
        print(f"{tool.name}: {tool.description}")
        
        if hasattr(tool, 'inputSchema') and isinstance(tool.inputSchema, dict):
            schema = tool.inputSchema
            
            if 'properties' in schema:
                print("Parameters:")
                for param_name, param_info in schema['properties'].items():
                    param_type = param_info.get('type', 'string')
                    is_required = 'required' in schema and param_name in schema['required']
                    
                    print(f"  --{param_name} ({param_type})", end="")
                    if is_required:
                        print(" (required)", end="")
                    print()
                    
                    if 'description' in param_info:
                        print(f"    {param_info['description']}")
            
            if 'required' in schema and schema['required']:
                print(f"Required parameters: {', '.join(schema['required'])}")
        
        print("\nUsage:")
        print(f"  {self.get_example_usage(tool)}")

    def check_required_args(self, tool, args: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if all required arguments are provided for a tool."""
        if not hasattr(tool, 'inputSchema') or not isinstance(tool.inputSchema, dict):
            return True, []
        
        schema = tool.inputSchema
        if 'required' not in schema or not schema['required']:
            return True, []
        
        missing = []
        for req_param in schema['required']:
            if req_param not in args or args[req_param] is None:
                missing.append(req_param)
        
        return len(missing) == 0, missing

    def parse_command_line(self, argv: List[str]) -> Tuple[argparse.Namespace, Optional[str], Dict[str, Any]]:
        """Parse command line arguments."""
        # Parse global args first
        global_parser = argparse.ArgumentParser(add_help=False)
        global_parser.add_argument('--timeout', type=int, default=5, 
                                help='Connection timeout in seconds (default: 5)')
        global_parser.add_argument('--debug', action='store_true', 
                                help='Enable debug logging')
        global_parser.add_argument('--transport', choices=['sse', 'websocket'], default='sse',
                                help='Transport protocol to use: sse or websocket (default: sse)')
        
        # First pass to get global args and tool name
        global_args, remaining = global_parser.parse_known_args(argv)
        
        # Get tool name if provided
        tool_name = None
        tool_args = {}
        
        if remaining and not remaining[0].startswith('--'):
            tool_name = remaining[0]
            remaining = remaining[1:]
        
        # Parse remaining args as key-value pairs
        i = 0
        while i < len(remaining):
            if remaining[i].startswith('--'):
                param = remaining[i][2:]  # Remove '--'
                if i + 1 < len(remaining) and not remaining[i + 1].startswith('--'):
                    # Parameter with value
                    tool_args[param] = remaining[i + 1]
                    i += 2
                else:
                    # Flag parameter (boolean)
                    tool_args[param] = True
                    i += 1
            else:
                # Unrecognized argument format
                i += 1
        
        return global_args, tool_name, tool_args

    def convert_arg_types(self, tool, args: Dict[str, Any]) -> Dict[str, Any]:
        """Convert argument types based on tool schema."""
        if not hasattr(tool, 'inputSchema') or not isinstance(tool.inputSchema, dict):
            return args
        
        schema = tool.inputSchema
        if 'properties' not in schema:
            return args
        
        converted_args = {}
        for param_name, value in args.items():
            if param_name in schema['properties']:
                param_info = schema['properties'][param_name]
                param_type = param_info.get('type')
                
                try:
                    if param_type == 'integer':
                        converted_args[param_name] = int(value)
                    elif param_type == 'number':
                        converted_args[param_name] = float(value)
                    elif param_type == 'boolean':
                        if isinstance(value, str):
                            converted_args[param_name] = value.lower() in ('true', 'yes', 'y', '1')
                        else:
                            converted_args[param_name] = bool(value)
                    elif param_type == 'array':
                        if isinstance(value, str):
                            # Try to parse as JSON array first
                            try:
                                converted_args[param_name] = json.loads(value)
                            except json.JSONDecodeError:
                                # Fall back to space-separated values
                                converted_args[param_name] = value.split()
                        elif isinstance(value, list):
                            converted_args[param_name] = value
                        else:
                            converted_args[param_name] = [value]
                    elif param_type == 'object':
                        if isinstance(value, str):
                            converted_args[param_name] = json.loads(value)
                        else:
                            converted_args[param_name] = value
                    else:
                        # Default to string
                        converted_args[param_name] = value
                except (ValueError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to convert parameter {param_name} to type {param_type}: {e}")
                    converted_args[param_name] = value
            else:
                converted_args[param_name] = value
        
        return converted_args

    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool with the given arguments."""
        if not self.tools:
            raise ValueError("No tools discovered yet")
        
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        # Check required args
        valid, missing = self.check_required_args(tool, args)
        if not valid:
            missing_args = ", ".join(f"--{arg}" for arg in missing)
            raise ValueError(f"Missing required arguments for tool '{tool_name}': {missing_args}")
        
        # Convert argument types
        converted_args = self.convert_arg_types(tool, args)
        
        # Remove None values from args
        args_to_send = {k: v for k, v in converted_args.items() if v is not None}
        
        try:
            return await self.client.call_tool(tool_name, args_to_send)
        except asyncio.TimeoutError:
            raise ValueError(f"Connection timeout: Server did not respond within {self.client.timeout} seconds")

    async def run(self, argv: Optional[List[str]] = None) -> int:
        """Run the CLI with the given arguments."""
        if argv is None:
            argv = sys.argv[1:]
        
        # Parse command line
        global_args, tool_name, tool_args = self.parse_command_line(argv)
        
        # Set up logging
        if global_args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Update client with global args
        url = os.environ.get("MCP_HOST", "http://localhost:8000")
        self.client = MCPToolDiscoveryClient(url, global_args.timeout, global_args.transport)
        
        try:
            # Discover tools
            try:
                await self.discover_tools()
            except asyncio.TimeoutError as e:
                print(f"Error: {str(e)}")
                return 1
            except ConnectionError as e:
                # Handle the more specific connection errors
                print(f"Error: {str(e)}")
                print("\nPossible solutions:")
                print("1. Check if the hostname and port are correct")
                print("2. Ensure the MCP server is running at the specified URL")
                print(f"3. Try a different transport protocol (current: {global_args.transport})")
                print("4. Check your network connection")
                print("\nYou can start the server with: python start_mcp_server.py --transport", global_args.transport)
                return 1
            except Exception as e:
                # Get the most detailed error message possible
                if hasattr(e, '__context__') and e.__context__ is not None:
                    error_msg = str(e.__context__)
                else:
                    error_msg = str(e)
                
                print(f"Error connecting to server at {url}: {error_msg}")
                
                # If it's a connection error, provide a more helpful message
                if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                    print(f"\nMake sure the MCP server is running at {url}")
                    print("You can start the server with: python start_mcp_server.py")
                
                return 1
            
            if not tool_name:
                # No tool specified, show all tools with examples
                self.display_all_tools()
                return 0
            
            # Check if the tool exists
            if not self.tools:
                print("No tools available.")
                return 1
                
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                print(f"Tool '{tool_name}' not found.")
                print("\nAvailable tools:")
                for t in self.tools:
                    print(f"  {t.name}: {t.description}")
                return 1
            
            # Check if all required arguments are provided
            valid, missing = self.check_required_args(tool, tool_args)
            
            if not valid:
                # Missing required arguments, show usage
                print(f"Missing required arguments for tool '{tool_name}': {', '.join(missing)}")
                self.display_tool_usage(tool_name)
                return 1
            
            # Execute the tool
            try:
                result = await self.execute_tool(tool_name, tool_args)
                if isinstance(result, (dict, list)):
                    print(json.dumps(result, indent=2))
                else:
                    print(result)
                return 0
            except ValueError as e:
                # Handle timeout and other value errors without traceback
                print(f"Error: {str(e)}")
                return 1
            except asyncio.TimeoutError:
                print(f"Error: Connection timeout - Server at {url} did not respond within {global_args.timeout} seconds")
                print(f"Tip: Try using a different transport protocol with --transport (current: {global_args.transport})")
                return 1
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return 1
                
        except ValueError as e:
            # Handle timeout converted to value error and other value errors
            print(f"Error: {str(e)}")
            return 1
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return 1

async def main() -> int:
    cli = MCPCliDynamic()
    return await cli.run()

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        logger.info("Client terminated by user")
        sys.exit(130)  # 130 is the standard exit code for SIGINT
    except asyncio.TimeoutError as e:
        print(f"Error: {str(e)}")
        print("Tip: Try using a different transport protocol with --transport")
        sys.exit(1)
    except ConnectionError as e:
        print(f"Error: {str(e)}")
        print("\nPossible solutions:")
        print("1. Check if the hostname and port are correct")
        print("2. Ensure the MCP server is running at the specified URL")
        print("3. Try a different transport protocol (use --transport sse or --transport websocket)")
        print("4. Check your network connection")
        sys.exit(1)
    except Exception as e:
        # Get the most detailed error message possible
        if hasattr(e, '__context__') and e.__context__ is not None:
            error_msg = str(e.__context__)
        else:
            error_msg = str(e)
        
        print(f"Error: {error_msg}")
        sys.exit(1)
