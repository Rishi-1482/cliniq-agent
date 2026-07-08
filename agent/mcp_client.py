"""Manages the connection to the ClinIQ MCP server.

Provides a shared MCP client factory and a helper for unpacking tool results.
"""
import json
from pathlib import Path
from typing import Any

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


def unpack_mcp_result(result: Any) -> Any:
    """Unpack a langchain-mcp-adapters tool result into a plain Python value.

    In langchain-mcp-adapters 0.3.x, tool results come back as a list (or
    single item) of TextContent-shaped dicts:
        [{"type": "text", "text": "<json string>", "id": "..."}, ...]

    This helper strips the wrapper and JSON-decodes each `text` field so
    callers get plain Python values matching what the MCP tool returned.
    """
    if isinstance(result, list):
        return [
            json.loads(item["text"])
            if isinstance(item, dict) and item.get("type") == "text"
            else item
            for item in result
        ]
    if isinstance(result, dict) and result.get("type") == "text":
        return json.loads(result["text"])
    return result