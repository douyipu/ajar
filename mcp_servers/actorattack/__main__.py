"""ActorAttack MCP Server entry point

Usage: python -m mcp_servers.actorattack

Environment variables (by priority):
- ACTORATTACK_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY: API Key
- ACTORATTACK_BASE_URL / DEEPSEEK_BASE_URL: API Base URL (required for DeepSeek)
- ACTORATTACK_MODEL: Model name (default: deepseek-chat)
- ACTORATTACK_SESSION_ID: Session ID for state (default: default)
"""

from .server import server


def main():
    server.run()


if __name__ == "__main__":
    main()
