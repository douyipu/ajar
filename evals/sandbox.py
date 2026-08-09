"""Auditor helper tools: E2B sandbox code execution and Tavily web search."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from inspect_ai.tool import Tool, tool

if TYPE_CHECKING:
    from e2b_code_interpreter import Sandbox

_sandbox_cache: dict[str | None, Sandbox] = {}


def _get_or_create_sandbox_sync(instance: str | None = None) -> Sandbox:
    from e2b_code_interpreter import Sandbox

    if instance in _sandbox_cache:
        sbx = _sandbox_cache[instance]
        try:
            sbx.files.list("/")
            return sbx
        except Exception:
            try:
                sbx.kill()
            except Exception:
                pass
            _sandbox_cache.pop(instance, None)

    sbx = Sandbox.create()
    _sandbox_cache[instance] = sbx
    return sbx


def _run_code_in_sandbox_sync(code: str, instance: str | None = None) -> str:
    sbx = _get_or_create_sandbox_sync(instance)

    try:
        execution = sbx.run_code(code)
        result_parts = []

        if execution.logs.stdout:
            stdout = (
                execution.logs.stdout
                if isinstance(execution.logs.stdout, str)
                else "".join(execution.logs.stdout)
            )
            if stdout.strip():
                result_parts.append(stdout.strip())

        if execution.logs.stderr:
            stderr = (
                execution.logs.stderr
                if isinstance(execution.logs.stderr, str)
                else "".join(execution.logs.stderr)
            )
            if stderr.strip():
                result_parts.append(f"[stderr]\n{stderr.strip()}")

        if execution.error:
            error_msg = f"{execution.error.name}: {execution.error.value}"
            if execution.error.traceback:
                error_msg = (
                    f"Traceback (most recent call last):\n"
                    f"{execution.error.traceback}\n{error_msg}"
                )
            result_parts.append(error_msg)

        if not result_parts:
            return "(Code executed successfully with no output)"
        return "\n".join(result_parts)

    except Exception as e:
        if instance in _sandbox_cache:
            try:
                _sandbox_cache[instance].kill()
            except Exception:
                pass
            _sandbox_cache.pop(instance, None)
        return f"Sandbox execution error: {str(e)}"


def cleanup_sandbox(instance: str | None = None) -> None:
    if instance in _sandbox_cache:
        try:
            _sandbox_cache[instance].kill()
        except Exception:
            pass
        _sandbox_cache.pop(instance, None)


def cleanup_all_sandboxes() -> None:
    for instance in list(_sandbox_cache.keys()):
        cleanup_sandbox(instance)


@tool(name="execute_python")
def execute_code_in_sandbox(instance: str | None = None) -> Tool:
    async def execute(code: str) -> str:
        """
        Execute Python code in a real e2b sandbox and return the result.

        Use when the target requests Python execution via a tool call. After
        getting the result, send it back with `send_tool_call_result` then `resume`.

        Args:
            code: The Python code to execute
        """
        return await asyncio.to_thread(_run_code_in_sandbox_sync, code, instance)

    return execute


def _web_search_sync(query: str, max_results: int = 5) -> str:
    import os

    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is not set."

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=max_results)
        results = response.get("results", [])
        if not results:
            return f"No results found for query: {query}"

        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No content")
            formatted_results.append(f"[{i}] {title}\n    URL: {url}\n    {content}")
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Web search error: {str(e)}"


@tool(name="web_search")
def execute_web_search() -> Tool:
    async def execute(query: str, max_results: int = 5) -> str:
        """
        Perform a real web search using Tavily API.

        After getting results, send them with `send_tool_call_result` then `resume`.

        Args:
            query: The search query string
            max_results: Maximum number of results to return (default: 5)
        """
        return await asyncio.to_thread(_web_search_sync, query, max_results)

    return execute
