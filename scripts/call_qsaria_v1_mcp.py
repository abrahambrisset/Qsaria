#!/usr/bin/env python3
"""Use Qsaria's actual stdio MCP server with a repository-local test workspace."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / ".files" / "publication" / "agentic"
SERVER_ARGS = [
    "--profile",
    "qsaria",
    "--llm-policy",
    "disabled",
    "--no-chatgpt-compat",
    "--no-prompts",
    "--no-resources",
]


def server_parameters(workspace: Path = WORKSPACE) -> StdioServerParameters:
    paths = {
        name: workspace / directory
        for name, directory in (
            ("CS_COPILOT_STORAGE_ROOT", "storage"),
            ("HF_HOME", "models"),
            ("MPLCONFIGDIR", "mpl"),
            ("TMPDIR", "tmp"),
            ("XDG_CACHE_HOME", "cache"),
        )
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        **{name: str(path) for name, path in paths.items()},
        "QSARIA_MODEL_CATALOG_PATH": str(workspace / "catalog.json"),
        "USE_S3": "false",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "AGNO_TELEMETRY": "false",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1",
        "OMP_THREAD_LIMIT": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CS_COPILOT_DISABLE_PROCESS_POOLS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if sys.platform == "darwin":
        libomp = (
            Path(sys.prefix)
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
            / "torch"
            / "lib"
        )
        if (libomp / "libomp.dylib").is_file():
            environment["DYLD_FALLBACK_LIBRARY_PATH"] = str(libomp)
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "cs_copilot.mcp", *SERVER_ARGS],
        cwd=ROOT,
        env=environment,
    )


def payload(response):
    value = response.structuredContent
    if value is None and len(response.content) == 1 and response.content[0].type == "text":
        value = response.content[0].text
    if isinstance(value, dict) and value.keys() == {"result"}:
        value = value["result"]
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


async def call(args: argparse.Namespace) -> int:
    async with (
        asyncio.timeout(70),
        stdio_client(server_parameters()) as (reader, writer),
        ClientSession(reader, writer) as session,
    ):
        await session.initialize()
        if args.tool == "list":
            tools = await session.list_tools()
            print(json.dumps([tool.name for tool in tools.tools], indent=2))
            return 0
        if args.tool == "describe":
            tools = await session.list_tools()
            tool = next(tool for tool in tools.tools if tool.name == args.arguments)
            print(tool.model_dump_json(indent=2, exclude_none=True))
            return 0
        arguments = json.loads(args.arguments)
        response = await session.call_tool(args.tool, arguments)
    evidence = WORKSPACE / "tool_calls"
    evidence.mkdir(exist_ok=True)
    path = evidence / f"{uuid4().hex}.json"
    result = payload(response)
    path.write_text(
        json.dumps(
            {
                "tool": args.tool,
                "arguments": arguments,
                "is_error": response.isError,
                "result": result,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Evidence: {path}", file=sys.stderr)
    if args.select and isinstance(result, dict) and not response.isError:
        result = {name: result.get(name) for name in args.select.split(",")}
    print(json.dumps(result, indent=2))
    return int(response.isError)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", help="MCP tool name, list, or describe")
    parser.add_argument("arguments", nargs="?", default="{}")
    parser.add_argument(
        "--select", help="Comma-separated fields to print from the complete evidence"
    )
    raise SystemExit(asyncio.run(call(parser.parse_args())))


if __name__ == "__main__":
    main()
