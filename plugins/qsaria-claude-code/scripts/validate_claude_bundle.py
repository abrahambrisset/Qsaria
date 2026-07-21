#!/usr/bin/env python3
"""Validate the static Qsaria Claude Code plugin bundle without starting MCP."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
MARKETPLACE_PATH = REPO_ROOT / ".claude-plugin" / "marketplace.json"
PROJECT_SETTINGS_PATH = REPO_ROOT / ".claude" / "settings.json"
CODEX_CONTRACT_ROOT = REPO_ROOT / "plugins" / "qsaria-codex" / "contracts"
TOOL_PREFIX = "mcp__plugin_qsaria-claude-code_qsaria__"

ROLE_AGENTS = {
    "qsaria_curation": ("qsaria-curation", "qsaria-curate"),
    "qsaria_training": ("qsaria-training", "qsaria-train"),
    "qsaria_registry": ("qsaria-registry", "qsaria-registry"),
    "qsaria_inference": ("qsaria-inference", "qsaria-infer"),
    "qsaria_report": ("qsaria-report", "qsaria-report"),
}
LIFECYCLE_TOOLS = {
    "qsaria_create_experiment",
    "qsaria_list_experiments",
    "qsaria_complete_experiment",
}
COORDINATOR_BUILTINS = {"AskUserQuestion", "WebFetch", "WebSearch"}
FORBIDDEN_TOOLS = {
    "Bash",
    "Edit",
    "Write",
    "Read",
    "Skill",
    "SendMessage",
    "Task",
}
FORBIDDEN_AGENT_FIELDS = {
    "hooks",
    "mcpServers",
    "permissionMode",
    "memory",
    "isolation",
}


class BundleError(RuntimeError):
    """Raised when the Claude bundle violates its static contract."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BundleError(f"expected a JSON object at {path}")
    return payload


def _frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise BundleError(f"missing YAML frontmatter at {path}")
    try:
        raw, body = text[4:].split("\n---\n", 1)
    except ValueError as exc:
        raise BundleError(f"unterminated YAML frontmatter at {path}") from exc
    try:
        metadata = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise BundleError(f"invalid YAML at {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise BundleError(f"frontmatter must be an object at {path}")
    return metadata, body


def _normalize_handoff(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    normalized.pop("$id", None)
    return normalized


def _validate_manifest(errors: list[str]) -> None:
    manifest = _load_json(PLUGIN_ROOT / ".claude-plugin" / "plugin.json")
    expected = {
        "name": "qsaria-claude-code",
        "displayName": "Qsaria for Claude Code",
        "version": "0.2.0",
        "skills": "./skills/",
        "mcpServers": "./.mcp.json",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            errors.append(f"plugin manifest {key!r} must be {value!r}")
    if "agents" in manifest:
        errors.append("plugin agents must use Claude's default agents/ discovery")
    if any(key in manifest for key in ("hooks", "commands", "lspServers")):
        errors.append("plugin manifest exposes an unplanned executable component")

    settings = _load_json(PLUGIN_ROOT / "settings.json")
    if settings != {"agent": "qsaria-coordinator"}:
        errors.append("settings.json must activate only qsaria-coordinator")


def _validate_marketplace(errors: list[str]) -> None:
    marketplace = _load_json(MARKETPLACE_PATH)
    if marketplace.get("name") != "personal":
        errors.append("Claude marketplace must be named personal")
    entries = [
        item
        for item in marketplace.get("plugins", [])
        if isinstance(item, dict) and item.get("name") == "qsaria-claude-code"
    ]
    if len(entries) != 1:
        errors.append("personal marketplace must contain one qsaria-claude-code entry")
        return
    entry = entries[0]
    if entry.get("source") != "./plugins/qsaria-claude-code":
        errors.append("Claude plugin source must stay repository-local")
    if entry.get("strict") is not True or entry.get("defaultEnabled") is not True:
        errors.append("Claude marketplace entry must be strict and enabled after install")
    serialized = json.dumps(marketplace).lower()
    if "claude-community" in serialized or "claude-plugins-official" in serialized:
        errors.append("personal marketplace must not declare a public Anthropic destination")


def _validate_project_activation(errors: list[str]) -> None:
    settings = _load_json(PROJECT_SETTINGS_PATH)
    expected = {"enabledPlugins": {"qsaria-claude-code@personal": True}}
    if settings != expected:
        errors.append("project settings must enable only qsaria-claude-code@personal")


def _validate_mcp(errors: list[str]) -> None:
    payload = _load_json(PLUGIN_ROOT / ".mcp.json")
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict) or set(servers) != {"qsaria"}:
        errors.append(".mcp.json must expose exactly the qsaria server")
        return
    server = servers["qsaria"]
    args = server.get("args")
    if server.get("type") != "stdio" or server.get("command") != "uv":
        errors.append("Qsaria MCP must use uv over stdio")
    expected_prefix = [
        "--directory",
        "${CLAUDE_PROJECT_DIR}",
        "run",
        "--no-sync",
        "cscopilot-mcp",
    ]
    if not isinstance(args, list) or args[:5] != expected_prefix:
        errors.append("Qsaria MCP must run from CLAUDE_PROJECT_DIR with uv --no-sync")
        return
    required_args = {
        "--profile",
        "qsaria",
        "--llm-policy",
        "disabled",
        "--no-chatgpt-compat",
        "--no-prompts",
        "--no-resources",
    }
    if not required_args <= set(args):
        errors.append("Qsaria MCP launcher is missing isolation arguments")
    if server.get("timeout") != 3_600_000:
        errors.append("Claude MCP timeout must be 3,600,000 milliseconds")
    if any(key in server for key in ("tool_timeout_sec", "startup_timeout_sec", "required")):
        errors.append("Codex-only MCP fields must not appear in the Claude launcher")
    serialized = json.dumps(server)
    if "jobs.py" in serialized or "worker.py" in serialized:
        errors.append("ChEMBL job runner files must not appear in the Qsaria launcher")


def _validate_contracts(errors: list[str]) -> dict[str, Any]:
    contract = _load_json(PLUGIN_ROOT / "contracts" / "agent-tools.json")
    codex_contract = _load_json(CODEX_CONTRACT_ROOT / "agent-tools.json")
    if contract != codex_contract:
        errors.append("Claude agent-tools contract has drifted from Codex")

    handoff = _load_json(PLUGIN_ROOT / "contracts" / "handoff.schema.json")
    codex_handoff = _load_json(CODEX_CONTRACT_ROOT / "handoff.schema.json")
    if _normalize_handoff(handoff) != _normalize_handoff(codex_handoff):
        errors.append("Claude handoff schema has drifted from Codex")

    role_tools = contract.get("roles", {})
    if set(role_tools) != set(ROLE_AGENTS):
        errors.append("agent-tools contract must define exactly the five Qsaria roles")
    all_tools = {tool for tools in role_tools.values() for tool in tools}
    all_tools.update(LIFECYCLE_TOOLS)
    if len(all_tools) != 46:
        errors.append(f"expected 46 Qsaria tools, found {len(all_tools)}")
    return contract


def _validate_agent_common(path: Path, metadata: dict[str, Any], errors: list[str]) -> None:
    forbidden_fields = FORBIDDEN_AGENT_FIELDS.intersection(metadata)
    if forbidden_fields:
        errors.append(f"{path.name} uses unsupported fields: {sorted(forbidden_fields)}")
    tools = metadata.get("tools")
    if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
        errors.append(f"{path.name} tools must be an explicit string list")
        return
    forbidden_tools = FORBIDDEN_TOOLS.intersection(tools)
    if forbidden_tools:
        errors.append(f"{path.name} exposes forbidden tools: {sorted(forbidden_tools)}")


def _validate_agents(contract: dict[str, Any], errors: list[str]) -> None:
    agent_root = PLUGIN_ROOT / "agents"
    actual_names = {path.stem for path in agent_root.glob("*.md")}
    expected_names = {"qsaria-coordinator"} | {agent_name for agent_name, _ in ROLE_AGENTS.values()}
    if actual_names != expected_names:
        errors.append(f"unexpected Claude agent inventory: {sorted(actual_names)}")

    coordinator_path = agent_root / "qsaria-coordinator.md"
    coordinator, coordinator_body = _frontmatter(coordinator_path)
    _validate_agent_common(coordinator_path, coordinator, errors)
    if coordinator.get("name") != "qsaria-coordinator":
        errors.append("coordinator name is invalid")
    if coordinator.get("model") != "inherit" or "effort" in coordinator:
        errors.append("coordinator must inherit the user-selected model and effort")
    if coordinator.get("skills") != ["qsaria-claude-code:qsaria-coordinate"]:
        errors.append("coordinator must preload only qsaria-coordinate")

    coordinator_tools = set(coordinator.get("tools", []))
    expected_agent_tool = (
        "Agent(qsaria-claude-code:qsaria-curation, "
        "qsaria-claude-code:qsaria-training, "
        "qsaria-claude-code:qsaria-registry, "
        "qsaria-claude-code:qsaria-inference, "
        "qsaria-claude-code:qsaria-report)"
    )
    role_tools = {tool for tools in contract.get("roles", {}).values() for tool in tools}
    expected_mcp = {TOOL_PREFIX + tool for tool in role_tools | LIFECYCLE_TOOLS}
    expected_coordinator_tools = expected_mcp | COORDINATOR_BUILTINS | {expected_agent_tool}
    if coordinator_tools != expected_coordinator_tools:
        missing = sorted(expected_coordinator_tools - coordinator_tools)
        extra = sorted(coordinator_tools - expected_coordinator_tools)
        errors.append(f"coordinator tool mismatch; missing={missing}, extra={extra}")
    if "Never patch code" not in coordinator_body:
        errors.append("coordinator body must explicitly forbid maintenance")

    for role, (agent_name, skill_name) in ROLE_AGENTS.items():
        path = agent_root / f"{agent_name}.md"
        metadata, body = _frontmatter(path)
        _validate_agent_common(path, metadata, errors)
        if metadata.get("name") != agent_name:
            errors.append(f"{path.name} has the wrong agent name")
        if metadata.get("model") != "sonnet" or metadata.get("effort") != "medium":
            errors.append(f"{path.name} must use Sonnet medium")
        if metadata.get("maxTurns") != 30 or metadata.get("background") is not False:
            errors.append(f"{path.name} must be bounded to 30 foreground turns")
        expected_skill = [f"qsaria-claude-code:{skill_name}"]
        if metadata.get("skills") != expected_skill:
            errors.append(f"{path.name} must preload only {skill_name}")
        expected_tools = {TOOL_PREFIX + tool for tool in contract["roles"][role]}
        actual_tools = set(metadata.get("tools", []))
        if actual_tools != expected_tools:
            missing = sorted(expected_tools - actual_tools)
            extra = sorted(actual_tools - expected_tools)
            errors.append(f"{path.name} tool mismatch; missing={missing}, extra={extra}")
        normalized_body = " ".join(body.split())
        if "exactly one fresh public handoff" not in normalized_body:
            errors.append(f"{path.name} must require one fresh public handoff")


def _validate_skills(errors: list[str]) -> None:
    expected = {
        "qsaria-coordinate",
        "qsaria-curate",
        "qsaria-train",
        "qsaria-registry",
        "qsaria-infer",
        "qsaria-report",
    }
    skill_root = PLUGIN_ROOT / "skills"
    actual = {path.parent.name for path in skill_root.glob("*/SKILL.md")}
    if actual != expected:
        errors.append(f"unexpected Claude skill inventory: {sorted(actual)}")
    for name in sorted(expected):
        path = skill_root / name / "SKILL.md"
        metadata, body = _frontmatter(path)
        if metadata.get("name") != name:
            errors.append(f"{name} frontmatter name is invalid")
        if not isinstance(metadata.get("description"), str):
            errors.append(f"{name} needs a triggering description")
        if metadata.get("user-invocable") is not False:
            errors.append(f"{name} must be hidden from direct user invocation")
        if metadata.get("disable-model-invocation") is True:
            errors.append(f"{name} must remain preloadable by its agent")
        if "](references/" in body or "../../contracts/" in body:
            errors.append(f"{name} must be self-contained without filesystem Read")
        if len(body.splitlines()) > 220:
            errors.append(f"{name} exceeds the concise skill budget")


def _validate_absence(errors: list[str]) -> None:
    forbidden = [
        PLUGIN_ROOT / "hooks",
        PLUGIN_ROOT / "CLAUDE.md",
        PLUGIN_ROOT / ".lsp.json",
        PLUGIN_ROOT / "monitors",
    ]
    for path in forbidden:
        if path.exists():
            errors.append(f"unplanned Claude component exists: {path}")


def validate() -> list[str]:
    errors: list[str] = []
    try:
        _validate_manifest(errors)
        _validate_marketplace(errors)
        _validate_project_activation(errors)
        _validate_mcp(errors)
        contract = _validate_contracts(errors)
        _validate_agents(contract, errors)
        _validate_skills(errors)
        _validate_absence(errors)
    except BundleError as exc:
        errors.append(str(exc))
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Qsaria Claude Code bundle validation: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Qsaria Claude Code bundle validation: PASS")
    print("- personal marketplace: valid")
    print("- plugin components: 6 agents, 6 skills, 1 MCP server")
    print("- specialist tool allowlists: exact")
    print("- shared Codex contracts: synchronized")
    print("- hooks, memory, filesystem tools, and maintenance tools: absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
