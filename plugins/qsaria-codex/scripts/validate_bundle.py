#!/usr/bin/env python3
"""Validate the static Qsaria for Codex bundle and its role boundaries."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]

EXPECTED_SKILLS = {
    "qsaria-coordinate",
    "qsaria-curate",
    "qsaria-train",
    "qsaria-registry",
    "qsaria-infer",
    "qsaria-report",
}
EXPECTED_ROLES = {
    "qsaria_curation",
    "qsaria_training",
    "qsaria_registry",
    "qsaria_inference",
    "qsaria_report",
}
ROLE_SKILLS = {
    "qsaria_curation": "$qsaria-curate",
    "qsaria_training": "$qsaria-train",
    "qsaria_registry": "$qsaria-registry",
    "qsaria_inference": "$qsaria-infer",
    "qsaria_report": "$qsaria-report",
}
LIFECYCLE_AGENT_TOOLS = {
    "qsaria_bootstrap",
    "qsaria_open_experiment",
    "qsaria_get_experiment_state",
    "qsaria_list_artifacts",
    "qsaria_get_artifact",
    "qsaria_record_handoff",
}
COORDINATOR_ONLY_TOOLS = {
    "qsaria_create_experiment",
    "qsaria_list_experiments",
    "qsaria_complete_experiment",
}
REPORT_TOOLS = {"qsaria_report_build_context", "qsaria_report_save"}
STATUS_VALUES = {
    "completed",
    "partial",
    "retryable_error",
    "terminal_failure",
    "needs_user_input",
}


class Validation:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def read_json(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.errors.append(f"invalid JSON {path}: {exc}")
            return {}
        if not isinstance(payload, dict):
            self.errors.append(f"expected a JSON object in {path}")
            return {}
        return payload

    def read_toml(self, path: Path) -> dict[str, Any]:
        try:
            return tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            self.errors.append(f"invalid TOML {path}: {exc}")
            return {}


def _frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"\A---\n(.*?)\n---\n", text, flags=re.DOTALL)
    if match is None:
        return {}
    values: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _scientific_spec_names(repo_root: Path) -> set[str]:
    path = repo_root / "src/cs_copilot/mcp/tool_specs/qsaria.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "_spec":
            continue
        keywords = {item.arg: item.value for item in node.keywords if item.arg}
        surface = keywords.get("surface")
        method = keywords.get("method")
        if (
            isinstance(surface, ast.Constant)
            and isinstance(surface.value, str)
            and isinstance(method, ast.Constant)
            and isinstance(method.value, str)
        ):
            names.add(f"qsaria_{surface.value}_{method.value}")
    return names


def _lifecycle_spec_names(repo_root: Path) -> set[str]:
    path = repo_root / "src/cs_copilot/mcp/tool_specs/qsaria_lifecycle.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "_spec":
            continue
        name = next((item.value for item in node.keywords if item.arg == "name"), None)
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            names.add(name.value)
    return names


def validate_manifest(validation: Validation) -> None:
    manifest = validation.read_json(PLUGIN_ROOT / ".codex-plugin" / "plugin.json")
    validation.require(manifest.get("name") == "qsaria-codex", "wrong plugin technical name")
    validation.require(manifest.get("skills") == "./skills/", "manifest must expose ./skills/")
    validation.require(
        manifest.get("mcpServers") == "./.mcp.json", "manifest must expose ./.mcp.json"
    )
    interface = manifest.get("interface") or {}
    validation.require(
        interface.get("displayName") == "Qsaria for Codex", "wrong plugin display name"
    )
    validation.require(bool(interface.get("defaultPrompt")), "plugin starter prompts are missing")

    mcp = validation.read_json(PLUGIN_ROOT / ".mcp.json")
    servers = mcp.get("mcpServers") or {}
    validation.require(set(servers) == {"qsaria"}, "plugin must declare only the qsaria MCP server")
    server = servers.get("qsaria") or {}
    args = server.get("args") or []
    validation.require(server.get("command") == "uv", "Qsaria MCP must launch through uv")
    validation.require(
        args[:3] == ["run", "--no-sync", "cscopilot-mcp"],
        "Qsaria MCP must use uv run --no-sync cscopilot-mcp",
    )
    validation.require(
        "--profile" in args and args[args.index("--profile") + 1] == "qsaria",
        "Qsaria MCP profile is not selected",
    )
    validation.require(
        "--llm-policy" in args and args[args.index("--llm-policy") + 1] == "disabled",
        "Qsaria MCP LLM policy is not disabled",
    )
    validation.require(
        {"--no-chatgpt-compat", "--no-prompts", "--no-resources"} <= set(args),
        "Qsaria MCP must disable generic compatibility, prompts, and resources",
    )
    validation.require(server.get("tool_timeout_sec") == 3600, "MCP timeout must be 3600 seconds")
    forbidden = {"worker.py", "jobs.py", "agno", "team", "SESSION_ID", "USE_S3"}
    serialized = json.dumps(server)
    for token in forbidden:
        validation.require(
            token not in serialized, f"MCP declaration contains forbidden token {token}"
        )


def validate_runtime_plugin_version(validation: Validation, repo_root: Path) -> None:
    manifest = validation.read_json(PLUGIN_ROOT / ".codex-plugin" / "plugin.json")
    manifest_version = manifest.get("version")
    expected_base = manifest_version.split("+", 1)[0] if isinstance(manifest_version, str) else None
    runtime_path = repo_root / "src/cs_copilot/mcp/qsaria/contracts.py"
    try:
        tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        validation.errors.append(f"could not parse runtime plugin version in {runtime_path}: {exc}")
        return

    runtime_version: str | None = None
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "PLUGIN_CONTRACT_VERSION"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            runtime_version = node.value.value
    validation.require(
        runtime_version == expected_base,
        "runtime Qsaria plugin contract version differs from the manifest base version; "
        f"runtime={runtime_version!r}, manifest_base={expected_base!r}",
    )


def validate_skills(validation: Validation) -> None:
    skills_root = PLUGIN_ROOT / "skills"
    actual = {path.name for path in skills_root.iterdir() if path.is_dir()}
    validation.require(actual == EXPECTED_SKILLS, f"unexpected skill set: {sorted(actual)}")
    for skill_name in sorted(EXPECTED_SKILLS):
        root = skills_root / skill_name
        skill_path = root / "SKILL.md"
        try:
            text = skill_path.read_text(encoding="utf-8")
        except OSError as exc:
            validation.errors.append(f"could not read {skill_path}: {exc}")
            continue
        metadata = _frontmatter(text)
        validation.require(metadata.get("name") == skill_name, f"wrong skill name in {skill_path}")
        validation.require(
            len(metadata.get("description", "")) >= 40,
            f"skill description is incomplete in {skill_path}",
        )
        validation.require("TODO" not in text, f"TODO remains in {skill_path}")
        validation.require(
            "scientific-invariants.md" in text,
            f"{skill_name} does not load the shared scientific invariants",
        )
        validation.require(
            "handoff.schema.json" in text,
            f"{skill_name} does not load the structured handoff schema",
        )

        openai_path = root / "agents" / "openai.yaml"
        try:
            openai_text = openai_path.read_text(encoding="utf-8")
        except OSError as exc:
            validation.errors.append(f"could not read {openai_path}: {exc}")
            continue
        expected_implicit = "true" if skill_name == "qsaria-coordinate" else "false"
        pattern = rf"allow_implicit_invocation:\s*{expected_implicit}\b"
        validation.require(
            re.search(pattern, openai_text) is not None,
            f"wrong implicit-invocation policy for {skill_name}",
        )


def validate_handoff_contract(validation: Validation) -> None:
    schema = validation.read_json(PLUGIN_ROOT / "contracts" / "handoff.schema.json")
    required = set(schema.get("required") or [])
    expected_required = {
        "schema_version",
        "experiment_id",
        "agent",
        "status",
        "summary",
        "facts",
        "artifact_ids",
        "model_ids",
        "warnings",
        "blockers",
        "recommended_next_action",
    }
    validation.require(required == expected_required, "handoff required fields have drifted")
    properties = schema.get("properties") or {}
    agents = set(((properties.get("agent") or {}).get("enum") or []))
    statuses = set(((properties.get("status") or {}).get("enum") or []))
    validation.require(agents == EXPECTED_ROLES, "handoff agent enum has drifted")
    validation.require(statuses == STATUS_VALUES, "handoff status enum has drifted")
    validation.require(
        (properties.get("experiment_id") or {}).get("pattern")
        == r"^exp_[A-Za-z0-9][A-Za-z0-9._-]{2,95}$",
        "handoff experiment_id pattern has drifted",
    )
    validation.require(
        (properties.get("summary") or {}).get("minLength") == 1,
        "handoff summary must be non-empty",
    )
    validation.require(
        (properties.get("recommended_next_action") or {}).get("type") == "string",
        "handoff recommended_next_action must be a string",
    )


def validate_agents(validation: Validation, repo_root: Path) -> None:
    contract = validation.read_json(PLUGIN_ROOT / "contracts" / "agent-tools.json")
    roles = contract.get("roles") or {}
    validation.require(set(roles) == EXPECTED_ROLES, "agent-tools role set has drifted")
    validation.require(
        set(contract.get("common_read_tools") or [])
        == LIFECYCLE_AGENT_TOOLS - {"qsaria_record_handoff"},
        "agent-tools common read surface has drifted",
    )
    validation.require(
        contract.get("handoff_tool") == "qsaria_record_handoff",
        "wrong handoff tool in agent-tools contract",
    )

    try:
        scientific_names = _scientific_spec_names(repo_root)
        lifecycle_names = _lifecycle_spec_names(repo_root)
    except (OSError, SyntaxError) as exc:
        validation.errors.append(f"could not parse Qsaria SPECS: {exc}")
        scientific_names = set()
        lifecycle_names = set()
    validation.require(len(scientific_names) == 35, "expected 35 Qsaria scientific SPECS")
    expected_lifecycle = LIFECYCLE_AGENT_TOOLS | COORDINATOR_ONLY_TOOLS | REPORT_TOOLS
    validation.require(
        lifecycle_names == expected_lifecycle,
        "Qsaria lifecycle/report SPECS differ from the 11-tool contract",
    )
    validation.require(
        len(scientific_names | lifecycle_names) == 46,
        "expected an isolated 46-tool Qsaria profile",
    )
    expected_domain_by_role = {
        "qsaria_curation": {
            name for name in scientific_names if name.startswith("qsaria_curation_")
        },
        "qsaria_training": {
            name
            for name in scientific_names
            if name.startswith(("qsaria_training_", "qsaria_benchmark_", "qsaria_activity_cliffs_"))
        },
        "qsaria_registry": {
            name
            for name in scientific_names
            if name.startswith(("qsaria_registry_", "qsaria_ensemble_"))
        },
        "qsaria_inference": {
            name for name in scientific_names if name.startswith("qsaria_inference_")
        },
        "qsaria_report": REPORT_TOOLS,
    }

    templates_root = PLUGIN_ROOT / "assets" / "agents"
    templates = {path.stem: path for path in templates_root.glob("*.toml")}
    validation.require(set(templates) == EXPECTED_ROLES, "agent template set has drifted")

    exposed_scientific: set[str] = set()
    for role in sorted(EXPECTED_ROLES):
        configured = roles.get(role) or []
        validation.require(
            len(configured) == len(set(configured)), f"{role} allowlist contains duplicates"
        )
        validation.require(
            configured.count("qsaria_record_handoff") == 1,
            f"{role} must include the handoff tool exactly once",
        )
        validation.require(
            LIFECYCLE_AGENT_TOOLS <= set(configured),
            f"{role} does not expose the complete lifecycle/handoff surface",
        )
        validation.require(
            not (set(configured) & COORDINATOR_ONLY_TOOLS),
            f"{role} contains coordinator-only lifecycle tools",
        )
        domain_tools = set(configured) - LIFECYCLE_AGENT_TOOLS
        validation.require(
            domain_tools <= scientific_names | REPORT_TOOLS,
            f"{role} contains tools absent from tool_specs/qsaria.py or report facade: "
            f"{sorted(domain_tools - scientific_names - REPORT_TOOLS)}",
        )
        validation.require(
            domain_tools == expected_domain_by_role[role],
            f"{role} violates its role-specific MCP allowlist; "
            f"missing={sorted(expected_domain_by_role[role] - domain_tools)}; "
            f"extra={sorted(domain_tools - expected_domain_by_role[role])}",
        )
        exposed_scientific.update(domain_tools & scientific_names)

        path = templates.get(role)
        if path is None:
            continue
        template = validation.read_toml(path)
        validation.require(template.get("name") == role, f"wrong name in {path}")
        validation.require(template.get("model") == "gpt-5.6-terra", f"wrong model in {path}")
        validation.require(
            template.get("model_reasoning_effort") == "medium",
            f"wrong reasoning effort in {path}",
        )
        validation.require(template.get("sandbox_mode") == "read-only", f"wrong sandbox in {path}")
        instructions = template.get("developer_instructions") or ""
        validation.require(ROLE_SKILLS[role] in instructions, f"{path} does not select its skill")
        validation.require(
            instructions.count("qsaria_record_handoff") == 1,
            f"{path} must mention the handoff tool exactly once",
        )
        validation.require(
            "Never spawn" in instructions
            and re.search(r"do\s+not answer the user directly", instructions) is not None,
            f"{path} does not prohibit delegation and direct user answers",
        )
        validation.require(
            "mcp_servers" not in template, f"{path} uses forbidden top-level MCP config"
        )
        plugin_server = (
            ((template.get("plugins") or {}).get("qsaria-codex") or {})
            .get("mcp_servers", {})
            .get("qsaria", {})
        )
        validation.require(
            plugin_server.get("enabled") is True, f"plugin MCP is disabled in {path}"
        )
        validation.require(
            plugin_server.get("default_tools_approval_mode") == "auto",
            f"wrong MCP approval mode in {path}",
        )
        validation.require(
            plugin_server.get("enabled_tools") == configured,
            f"agent template allowlist differs from contract for {role}",
        )

    validation.require(
        exposed_scientific == scientific_names,
        "scientific SPECS not covered exactly by role allowlists; missing="
        f"{sorted(scientific_names - exposed_scientific)}",
    )

    project = validation.read_toml(PLUGIN_ROOT / "assets" / "project-config.toml")
    agents = project.get("agents") or {}
    validation.require(agents.get("max_depth") == 1, "project agent max_depth must be 1")
    validation.require(agents.get("max_threads") == 5, "project agent max_threads must be 5")
    validation.require(
        EXPECTED_ROLES <= set(agents), "project config does not register all Qsaria agents"
    )


def validate_misc(validation: Validation) -> None:
    required = {
        PLUGIN_ROOT / "README.md",
        PLUGIN_ROOT / "contracts" / "compatibility.json",
        PLUGIN_ROOT / "contracts" / "scientific-invariants.md",
        PLUGIN_ROOT / "scripts" / "preflight.py",
        PLUGIN_ROOT / "scripts" / "smoke_codex_plugin.py",
        PLUGIN_ROOT / "scripts" / "install_project_agents.py",
        PLUGIN_ROOT / "scripts" / "sync_qsaria_codex.py",
        PLUGIN_ROOT / "scripts" / "update_plugin_cachebuster.py",
    }
    for path in sorted(required):
        validation.require(path.is_file(), f"missing required bundle file {path}")
    for path in PLUGIN_ROOT.rglob("*"):
        if not path.is_file() or path.name == "compatibility.json":
            continue
        if path.suffix not in {".md", ".json", ".toml", ".yaml"}:
            continue
        validation.require("[TODO:" not in path.read_text(encoding="utf-8"), f"TODO in {path}")

    boundary_contract = (
        (PLUGIN_ROOT / "README.md").read_text(encoding="utf-8")
        + (PLUGIN_ROOT / "contracts" / "scientific-invariants.md").read_text(encoding="utf-8")
        + (PLUGIN_ROOT / "skills" / "qsaria-coordinate" / "SKILL.md").read_text(encoding="utf-8")
    )
    for token in (
        "QSARIA_MCP_ALLOWED_INPUT_ROOTS",
        "QSARIA_S3_SINGLE_WRITER=true",
        "single_writer_acknowledged=true",
        "configured bucket",
        "Path-based scientific",
        "uv sync --extra mcp --extra prediction",
        "--mcp-only",
    ):
        validation.require(
            token in boundary_contract, f"missing storage boundary contract: {token}"
        )


def validate_project_installation(validation: Validation, repo_root: Path) -> None:
    installer = PLUGIN_ROOT / "scripts" / "install_project_agents.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(installer),
            "--check",
            "--repo-root",
            str(repo_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    validation.require(
        completed.returncode == 0,
        "installed project agents differ from distribution templates: "
        + (completed.stderr.strip() or completed.stdout.strip()),
    )

    active_path = repo_root / ".codex" / "config.toml"
    template_path = PLUGIN_ROOT / "assets" / "project-config.toml"
    active = validation.read_toml(active_path)
    template = validation.read_toml(template_path)
    validation.require(
        active.get("agents") == template.get("agents"),
        ".codex/config.toml does not semantically match the Qsaria project template",
    )

    marketplace_path = repo_root / ".agents" / "plugins" / "marketplace.json"
    marketplace = validation.read_json(marketplace_path)
    validation.require(marketplace.get("name") == "personal", "wrong local marketplace name")
    entries = [
        item
        for item in marketplace.get("plugins") or []
        if isinstance(item, dict) and item.get("name") == "qsaria-codex"
    ]
    validation.require(len(entries) == 1, "marketplace must contain one Qsaria plugin entry")
    if len(entries) == 1:
        entry = entries[0]
        validation.require(
            entry.get("source") == {"source": "local", "path": "./plugins/qsaria-codex"},
            "marketplace points to the wrong Qsaria plugin source",
        )
        validation.require(
            entry.get("policy") == {"installation": "AVAILABLE", "authentication": "ON_INSTALL"},
            "marketplace has the wrong Qsaria installation policy",
        )
        validation.require(
            entry.get("category") == "Developer Tools",
            "marketplace has the wrong Qsaria category",
        )


def validate_compatibility(validation: Validation, repo_root: Path) -> None:
    command = [
        sys.executable,
        str(PLUGIN_ROOT / "scripts" / "sync_qsaria_codex.py"),
        "--check",
        "--repo-root",
        str(repo_root),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    validation.require(
        completed.returncode == 0,
        "compatibility signatures are stale: "
        + (completed.stderr.strip() or completed.stdout.strip()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Qsaria repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--skip-compatibility",
        action="store_true",
        help="skip signature comparison while initially recording metadata",
    )
    args = parser.parse_args()

    validation = Validation()
    repo_root = args.repo_root.resolve()
    validate_manifest(validation)
    validate_runtime_plugin_version(validation, repo_root)
    validate_skills(validation)
    validate_handoff_contract(validation)
    validate_agents(validation, repo_root)
    validate_misc(validation)
    validate_project_installation(validation, repo_root)
    if not args.skip_compatibility:
        validate_compatibility(validation, repo_root)

    if validation.errors:
        print("Qsaria Codex bundle validation failed:")
        for error in validation.errors:
            print(f"- {error}")
        return 1
    print("Qsaria Codex bundle validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
