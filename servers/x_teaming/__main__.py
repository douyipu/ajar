"""X-Teaming MCP Server entry point: python -m servers.x_teaming"""

from .server import server


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
