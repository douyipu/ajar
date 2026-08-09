"""ActorAttack MCP Server entry point: python -m servers.actorattack"""

from .server import server


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
