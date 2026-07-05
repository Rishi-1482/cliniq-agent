"""Manages the connection to the ClinIQ MCP server.

Provides a shared MCP client that graph nodes can call to invoke tools.
"""
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

_SERVER_PATH = str(Path(__file__).resolve().parent.parent / "mcp_server" / "server.py")


def build_mcp_client() -> MultiServerMCPClient:
    """Configure the MCP client to launch our server as a stdio subprocess."""
    return MultiServerMCPClient({
        "cliniq": {
            "command": "uv",
            "args": ["run", "python", _SERVER_PATH],
            "transport": "stdio",
        }
    })