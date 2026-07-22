#!/usr/bin/env python3
"""Validate the Qsaria Claude Science bundle and its live MCP contract."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SKILLS = {
    "qsaria-science-setup",
    "qsaria-science-coordinate",
    "qsaria-science-curate",
    "qsaria-science-train",
    "qsaria-science-registry",
    "qsaria-science-infer",
    "qsaria-science-report",
}
EXPECTED_PROFILES = {
    "QSARIA_CURATION": "curation",
    "QSARIA_TRAINING": "training",
    "QSARIA_REGISTRY": "registry",
    "QSARIA_INFERENCE": "inference",
}
OPERATION_TOOLS = {
    "qsaria_curation_start_operation",
    "qsaria_training_start_operation",
    "qsaria_registry_start_operation",
    "qsaria_inference_start_operation",
    "qsaria_list_operations",
    "qsaria_get_operation_state",
    "qsaria_get_operation_result",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"missing YAML frontmatter: {path}")
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def validate() -> list[str]:
    errors: list[str] = []
    if (PLUGIN_ROOT / ".claude-plugin").exists():
        errors.append("Claude Science bundle must not contain .claude-plugin")
    if not (PLUGIN_ROOT / "bundle.json").is_file():
        errors.append("missing bundle.json")
        bundle = {}
    else:
        bundle = _load_json(PLUGIN_ROOT / "bundle.json")
    if bundle.get("name") != "qsaria-claude-science":
        errors.append("wrong Science bundle name")
    if bundle.get("minimum_claude_science_version") != "0.1.21":
        errors.append("bundle must declare Claude Science 0.1.21 compatibility")
    if bundle.get("public_plugin") is not False:
        errors.append("Science integration must remain a private non-plugin bundle")
    for required_path in (
        PLUGIN_ROOT / "README.md",
        PLUGIN_ROOT / "contracts" / "compatibility.json",
        PLUGIN_ROOT / "scripts" / "preflight.py",
        PLUGIN_ROOT / "scripts" / "durability_probe.py",
        PLUGIN_ROOT / "scripts" / "sync_qsaria_science.py",
    ):
        if not required_path.is_file():
            errors.append(f"missing Science bundle component: {required_path.name}")
    skill_dirs = {path.name for path in (PLUGIN_ROOT / "skills").iterdir() if path.is_dir()}
    if skill_dirs != EXPECTED_SKILLS:
        errors.append(f"skill set mismatch: {sorted(skill_dirs)}")
    for name in sorted(EXPECTED_SKILLS):
        path = PLUGIN_ROOT / "skills" / name / "SKILL.md"
        try:
            metadata = _frontmatter(path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if metadata.get("name") != name:
            errors.append(f"wrong skill name in {path}")
        if not metadata.get("description"):
            errors.append(f"missing skill description in {path}")
        openai_path = path.parent / "agents" / "openai.yaml"
        if not openai_path.is_file():
            errors.append(f"missing skill interface metadata: {openai_path}")

    contract = _load_json(PLUGIN_ROOT / "contracts" / "agent-tools.json")
    from cs_copilot.mcp.qsaria.operations import allowed_operation_specs
    from cs_copilot.mcp.tools_registry import all_specs

    actual_tools = {spec.mcp_name for spec in all_specs(profile="qsaria")}
    if len(actual_tools) != 53:
        errors.append(
            f"expected 53 Qsaria tools (46 synchronous + 7 operation), got {len(actual_tools)}"
        )
    if not OPERATION_TOOLS.issubset(actual_tools):
        errors.append("live MCP operation tools are incomplete")
    synchronous_tools = actual_tools - OPERATION_TOOLS
    if len(synchronous_tools) != 46:
        errors.append(f"expected unchanged 46 synchronous tools, got {len(synchronous_tools)}")
    roles = contract.get("roles") or {}
    if set(roles) != {*EXPECTED_PROFILES, "QSARIA_REPORT"}:
        errors.append("Science role contract must define exactly five specialists")
    for profile, role in EXPECTED_PROFILES.items():
        entry = roles.get(profile) or {}
        tools = set(entry.get("tools") or [])
        if not tools.issubset(actual_tools):
            errors.append(f"{profile} references unknown tools")
        expected_operations = set(allowed_operation_specs(role))
        if set(entry.get("detached_operations") or []) != expected_operations:
            errors.append(f"{profile} detached operation allowlist drifted")
        pattern = re.compile(
            "^(?:" + "|".join(re.escape(tool) for tool in entry.get("tools") or []) + ")$"
        )
        if {tool for tool in actual_tools if pattern.fullmatch(tool)} != tools:
            errors.append(f"{profile} exact connector regex is unsafe")

    report_entry = roles.get("QSARIA_REPORT") or {}
    if report_entry.get("start_tool") is not None or report_entry.get("detached_operations") != []:
        errors.append("Report must remain synchronous in V1")
    handoff = _load_json(PLUGIN_ROOT / "contracts" / "handoff.schema.json")
    canonical_handoff = _load_json(
        REPO_ROOT / "plugins" / "qsaria-codex" / "contracts" / "handoff.schema.json"
    )
    handoff.pop("$id", None)
    canonical_handoff.pop("$id", None)
    if handoff != canonical_handoff:
        errors.append("Science handoff schema drifted from the canonical Qsaria contract")
    launcher = (PLUGIN_ROOT / "scripts" / "launch-mcp.sh").read_text(encoding="utf-8")
    for required in (
        "--profile qsaria",
        "--llm-policy disabled",
        "--no-chatgpt-compat",
        "--no-prompts",
        "--no-resources",
        "QSARIA_MODEL_CATALOG_PATH",
        "QSARIA_SCIENCE_ARTIFACT_ROOT",
        "USE_S3=false",
    ):
        if required not in launcher:
            errors.append(f"launcher missing {required!r}")
    readme = (PLUGIN_ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "read-only",
        "read-write",
        "60-second",
        "qsaria-science-setup",
        "durability_probe.py",
    ):
        if required not in readme:
            errors.append(f"README missing installation rule {required!r}")
    return errors


def main() -> int:
    try:
        errors = validate()
    except Exception as exc:  # noqa: BLE001 - validator must report stable failure
        print(f"Qsaria Claude Science validation failed: {exc}", file=sys.stderr)
        return 2
    if errors:
        print("Qsaria Claude Science bundle: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Qsaria Claude Science bundle: PASS")
    print("- skills: 7")
    print("- profiles: coordinator + 5 specialists")
    print("- deterministic tools: 46 synchronous + 7 durable-operation tools")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
