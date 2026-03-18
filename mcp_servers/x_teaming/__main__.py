"""X-Teaming MCP Server entry point

Usage: python -m mcp_servers.x_teaming

Environment variables (by priority):
- X_TEAMING_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY: API Key
- X_TEAMING_BASE_URL / DEEPSEEK_BASE_URL: API Base URL (required for DeepSeek)
- X_TEAMING_MODEL: Model name (default: deepseek-chat)
- X_TEAMING_SESSION_ID: Session ID for state (default: default)
"""

from .server import server


def main():
    server.run()


if __name__ == "__main__":
    main()
