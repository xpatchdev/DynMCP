#!/usr/bin/env python3
"""
MCP Tool Discovery Client - A client to connect to MCP servers and discover available tools.
This script demonstrates how to:
1. Connect to an MCP server using different transport types (SSE, WebSocket)
2. Initialize a session
3. Discover available tools
4. Handle server messages and notifications
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import timedelta
from typing import Any, Optional, Literal, Union
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp-discovery")

try:
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
    
    from mcp.shared.session import RequestResponder
    from mcp.types import (
        JSONRPCMessage,
        ServerRequest,
        ClientResult,
        ServerNotification,
        Tool as MCPTool,
    )
except ImportError as e:
    print(f"Error: MCP client libraries not available: {e}")
    print("Install with: pip install mcp-server-api")
    sys.exit(1)



class MCPToolDiscoveryClient:
    def __init__(self, url: str, timeout: int = 30, transport: str = "sse"):
        """Initialize the MCP Tool Discovery Client.
        
        Args:
            url: The URL of the MCP server
            timeout: Connection timeout in seconds
            transport: Transport type: "sse" or "websocket"
        """
        self.transport = transport
        
        # Parse the URL
        parsed_url = urlparse(url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        
        # Adjust URL based on transport type
        if transport == "sse" and not self.base_url.endswith("/sse"):
            self.url = f"{self.base_url.rstrip('/')}/sse"
        elif transport == "websocket" and not self.base_url.endswith("/ws"):
            # For WebSocket, we need to change the scheme from http to ws
            ws_scheme = "wss" if parsed_url.scheme == "https" else "ws"
            self.url = f"{ws_scheme}://{parsed_url.netloc}{parsed_url.path.rstrip('/')}/ws"
        else:
            self.url = self.base_url
        
        self.timeout = timeout
        self._tools: list[MCPTool] = []
        
    def _extract_real_error(self, e: BaseException) -> Exception:
        """Extract the real error from a possibly nested exception."""
        # First try the __context__ attribute
        if hasattr(e, '__context__') and e.__context__ is not None:
            return self._extract_real_error(e.__context__)
        
        # Then try __cause__ attribute
        if hasattr(e, '__cause__') and e.__cause__ is not None:
            return self._extract_real_error(e.__cause__)
        
        # Check if it's a TaskGroup error which contains sub-exceptions
        if "TaskGroup" in str(e) and hasattr(e, '__dict__'):
            for attr_name, attr_value in e.__dict__.items():
                if isinstance(attr_value, list) and attr_value:
                    for sub_exc in attr_value:
                        if isinstance(sub_exc, Exception):
                            return self._extract_real_error(sub_exc)
        
        # If we couldn't find a more specific error, return the original
        # Ensure we return an Exception, not just any BaseException
        return Exception(str(e)) if not isinstance(e, Exception) else e

    async def message_handler(
        self,
        message: Union[RequestResponder, ServerNotification, Exception]
    ) -> None:
        """Handle incoming messages from the server."""
        if isinstance(message, Exception):
            logger.error(f"Error in message handler: {message}")
            return

        if hasattr(message, 'method'):  # ServerNotification check
            logger.info(f"Received notification: {message}")
            return

        logger.debug(f"Received message: {message}")

    async def _get_transport_client(self):
        """Get the appropriate transport client based on the transport type."""
        if self.transport == "sse":
            logger.debug(f"Using SSE transport at {self.url}")
            return sse_client(self.url, timeout=self.timeout)
        else:
            # Fall back to SSE transport
            if self.transport != "sse":
                logger.warning(f"Transport '{self.transport}' not available, falling back to SSE")
            
            # Ensure URL has correct scheme and /sse suffix for SSE transport
            sse_url = self.url
            
            # If URL has ws:// or wss:// scheme, change to http:// or https://
            parsed_url = urlparse(sse_url)
            if parsed_url.scheme in ["ws", "wss"]:
                http_scheme = "https" if parsed_url.scheme == "wss" else "http"
                sse_url = f"{http_scheme}://{parsed_url.netloc}{parsed_url.path}"
            
            # Add /sse suffix if needed
            if not sse_url.endswith("/sse"):
                sse_url = f"{sse_url.rstrip('/')}/sse"
                
            logger.debug(f"Using SSE transport at {sse_url}")
            return sse_client(sse_url, timeout=self.timeout)

    async def discover_tools(self) -> list[MCPTool]:
        """Connect to the MCP server and discover available tools."""
        logger.info(f"Connecting to MCP server at {self.url} using {self.transport} transport")
        
        try:
            # Properly await the transport client coroutine
            client = await self._get_transport_client()
            async with client as (read_stream, write_stream):
                logger.info(f"Connection established using {self.transport} transport")
                
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.timeout),
                    message_handler=self.message_handler
                ) as session:
                    # Initialize the session
                    logger.info("Initializing session...")
                    await session.initialize()
                    logger.info("Session initialized successfully")

                    # List available tools
                    logger.info("Discovering tools...")
                    try:
                        tools_result = await session.list_tools()
                        self._tools = tools_result.tools
                        
                        # Print discovered tools with detailed schema info
                        logger.info(f"Discovered {len(self._tools)} tools:")
                        for tool in self._tools:
                            try:
                                tool_info = [f"  - {tool.name}: {tool.description}"]
                                
                                if hasattr(tool, 'inputSchema'):
                                    schema = tool.inputSchema
                                    if isinstance(schema, dict):
                                        if 'properties' in schema:
                                            tool_info.append("    Parameters:")
                                            for param_name, param_info in schema['properties'].items():
                                                param_str = f"      - {param_name}"
                                                if 'type' in param_info:
                                                    param_str += f" (type: {param_info['type']})"
                                                if 'title' in param_info:
                                                    param_str += f"\n        Title: {param_info['title']}"
                                                if 'default' in param_info:
                                                    param_str += f"\n        Default: {param_info['default']}"
                                                if 'anyOf' in param_info:
                                                    types = [t.get('type') for t in param_info['anyOf']]
                                                    param_str += f"\n        Types: {' | '.join(t for t in types if t)}"
                                                tool_info.append(param_str)
                                        
                                        if 'required' in schema:
                                            tool_info.append(f"    Required parameters: {', '.join(schema['required'])}")
                                
                                logger.info("\n".join(tool_info))
                            except Exception as e:
                                logger.error(f"Error printing tool info for {getattr(tool, 'name', 'unknown tool')}: {e}")
                                continue

                        return self._tools
                    except Exception as e:
                        logger.error(f"Error listing tools: {e}")
                        raise

        except asyncio.TimeoutError as e:
            logger.error(f"Connection timeout ({self.timeout}s) when connecting to {self.url}")
            # Re-raise timeout error without traceback
            raise asyncio.TimeoutError(f"Connection timeout ({self.timeout}s) when connecting to {self.url}")
        except Exception as e:
            # Extract the actual exception if it's hiding in TaskGroup
            real_error = self._extract_real_error(e)
            error_msg = str(real_error)
            
            logger.error(f"Error during tool discovery: {error_msg}")
            
            # Convert to more specific exceptions for better handling
            if "getaddrinfo" in error_msg or "Name or service not known" in error_msg:
                raise ConnectionError(f"Could not resolve hostname: {self.url}")
            elif "No route to host" in error_msg:
                raise ConnectionError(f"No route to host: {self.url}")
            elif "Connection refused" in error_msg or "[Errno 111] Connect call failed" in error_msg:
                raise ConnectionError(f"Connection refused: {self.url} - Server is not running or port is wrong")
            elif "TimeoutError" in error_msg:
                raise asyncio.TimeoutError(f"Connection timeout when connecting to {self.url}")
            elif "Cancelled by cancel scope" in error_msg:
                # This is likely a connection timeout or DNS resolution failure
                if "localhos:" in self.url and "localhost:" not in self.url:
                    # Common typo, provide a more helpful message
                    raise ConnectionError(f"Could not resolve hostname. Did you mean 'localhost' instead of 'localhos'?")
                else:
                    raise ConnectionError(f"Connection failed: Could not connect to {self.url}")
            else:
                # Check if it might be a transport protocol issue
                if self.transport != "sse":
                    logger.warning(f"Connection with {self.transport} transport failed. Will attempt to fallback to SSE in future requests.")
                    self.transport = "sse"  # Set fallback transport for future requests
                    
                raise ConnectionError(f"Failed to connect to {self.url}: {error_msg}")

    async def call_tool(self, tool_name: str, arguments: Optional[dict[str, Any]] = None) -> Any:
        """Call a discovered tool by name."""
        if not self._tools:
            raise RuntimeError("No tools discovered yet. Call discover_tools() first.")

        tool = next((t for t in self._tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        try:
            # Properly await the transport client coroutine
            client = await self._get_transport_client()
            async with client as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.timeout),
                    message_handler=self.message_handler
                ) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments or {})
                    return result
        except asyncio.TimeoutError:
            logger.error(f"Connection timeout ({self.timeout}s) when calling tool '{tool_name}'")
            # Re-raise timeout error with clear message but without traceback
            raise asyncio.TimeoutError(f"Connection timeout ({self.timeout}s) when calling tool '{tool_name}'")
        except Exception as e:
            # Extract the actual exception if it's hiding in TaskGroup
            real_error = self._extract_real_error(e)
            error_msg = str(real_error)
            
            logger.error(f"Error calling tool {tool_name}: {error_msg}")
            
            # Convert to more specific exceptions for better handling
            if "getaddrinfo" in error_msg or "Name or service not known" in error_msg:
                raise ConnectionError(f"Could not resolve hostname when calling tool '{tool_name}'")
            elif "No route to host" in error_msg:
                raise ConnectionError(f"No route to host when calling tool '{tool_name}'")
            elif "Connection refused" in error_msg:
                raise ConnectionError(f"Connection refused when calling tool '{tool_name}'")
            elif "TimeoutError" in error_msg:
                raise asyncio.TimeoutError(f"Connection timeout when calling tool '{tool_name}'")
            elif "Cancelled by cancel scope" in error_msg:
                # This is likely a connection timeout or DNS resolution failure
                if "localhos:" in self.url and "localhost:" not in self.url:
                    # Common typo, provide a more helpful message
                    raise ConnectionError(f"Could not resolve hostname. Did you mean 'localhost' instead of 'localhos'?")
                else:
                    raise ConnectionError(f"Connection failed: Could not connect to server")
            else:
                # Check if it might be a transport protocol issue 
                if self.transport != "sse":
                    logger.warning(f"Tool call with {self.transport} transport failed. Trying to fallback to SSE in future requests.")
                    self.transport = "sse"  # Set fallback transport for future calls
                    
                raise ConnectionError(f"Failed to call tool '{tool_name}': {error_msg}")

async def main():
    parser = argparse.ArgumentParser(description="MCP Tool Discovery Client")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="MCP server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Connection timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--transport",
        choices=["sse", "websocket"],
        default="sse",
        help="Transport type: sse or websocket (default: sse)"
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    client = MCPToolDiscoveryClient(args.url, args.timeout, args.transport)
    try:
        await client.discover_tools()
    except Exception as e:
        logger.error(f"Failed to discover tools: {e}")
        return 1
    return 0

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Client terminated by user")
    except asyncio.TimeoutError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
