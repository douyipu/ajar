"""Crescendo MCP Server entry point

Usage: python -m mcp_servers.crescendo

Environment variables (by priority):
- CRESCENDO_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY: API Key
- CRESCENDO_BASE_URL / DEEPSEEK_BASE_URL: API Base URL (required for DeepSeek)
- CRESCENDO_MODEL: Model name (default: deepseek-chat)
- CRESCENDO_SESSION_ID: Session ID for state (default: default)
"""

from .server import server


def main():
    server.run()


if __name__ == "__main__":
    main()
