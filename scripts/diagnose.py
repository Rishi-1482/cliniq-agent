import _bootstrap
import sys
print("sys.executable:", sys.executable)
print("sys.path:")
for p in sys.path:
    print(" ", p)
print()
try:
    import mcp_server
    print("mcp_server found at:", mcp_server.__file__)
except ImportError as e:
    print("mcp_server IMPORT FAILED:", e)