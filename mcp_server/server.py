"""FastMCP server for ClinIQ. Hello-world version with one trivial tool."""
from mcp.server.fastmcp import FastMCP
from mcp_server.tools.pubmed import search_pubmed

mcp = FastMCP("cliniq")


@mcp.tool()
def echo(text: str) -> str:
    """Echo back the provided text. Used for testing the server plumbing."""
    return f"echo: {text}"


if __name__ == "__main__":
    # Run as a stdio MCP server
    mcp.run()