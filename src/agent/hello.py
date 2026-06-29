"""Hello-world LangGraph agent that calls the ClinIQ MCP server."""
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Path to our MCP server entrypoint
SERVER_PATH = str(Path(__file__).parent.parent / "mcp_server" / "server.py")


async def main():
    # Configure the MCP client to launch our server as a subprocess
    client = MultiServerMCPClient(
        {
            "cliniq": {
                "command": "uv",
                "args": ["run", "python", SERVER_PATH],
                "transport": "stdio",
            }
        }
    )

    # Discover the tools the server exposes
    tools = await client.get_tools()
    print(f"Discovered {len(tools)} tool(s) from MCP server:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    # Build a minimal ReAct agent with those tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(llm, tools)

    # Run a query that should make the agent call the echo tool
    print("\nAsking agent: 'Use the echo tool to say hello back to me.'")
    result = await agent.ainvoke(
        {"messages": [("user", "Use the echo tool to say hello back to me.")]}
    )

    # Print the final assistant message
    final_message = result["messages"][-1]
    print(f"\nAgent's final response:\n{final_message.content}")

    # Print the full message trace
    print("\n--- Full message trace ---")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        print(f"\n[{i}] {msg_type}")
        if hasattr(msg, "content") and msg.content:
            print(f"    content: {msg.content!r}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"    tool_calls: {msg.tool_calls}")
        if hasattr(msg, "name") and msg.name:
            print(f"    tool name: {msg.name}")


if __name__ == "__main__":
    asyncio.run(main())