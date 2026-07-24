"""Static acceptance tests for the personal Qsaria Claude Code plugin."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = REPO_ROOT / "plugins" / "qsaria-claude-code"
SCRIPTS_ROOT = PLUGIN_ROOT / "scripts"
TOOL_PREFIX = "mcp__plugin_qsaria-claude-code_qsaria__"
ROLE_AGENTS = {
    "qsaria_curation": ("qsaria-curation", "qsaria-curate"),
    "qsaria_training": ("qsaria-training", "qsaria-train"),
    "qsaria_registry": ("qsaria-registry", "qsaria-registry"),
    "qsaria_inference": ("qsaria-inference", "qsaria-infer"),
    "qsaria_report": ("qsaria-report", "qsaria-report"),
}


def _run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / name), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _frontmatter(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    raw, body = text[4:].split("\n---\n", 1)
    metadata = yaml.safe_load(raw)
    assert isinstance(metadata, dict)
    return metadata, body


def test_bundle_validator_passes() -> None:
    result = _run_script("validate_claude_bundle.py")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_phase_b_compatibility_signatures_are_current() -> None:
    result = _run_script("sync_qsaria_claude.py", "--check")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    compatibility = json.loads(
        (PLUGIN_ROOT / "contracts" / "compatibility.json").read_text(encoding="utf-8")
    )
    assert compatibility["phase"] == "phase_b_integrated_bundle"
    assert compatibility["client_contract"] == "claude_code_v1"
    assert compatibility["coordinator_contract"] == "external_mcp_coordinator_v1"
    assert compatibility["supported_clients"] == [
        "codex_v1",
        "claude_code_v1",
        "claude_science_v1",
    ]
    assert compatibility["runtime_bootstrap_validation"] == ("automated_validate_shared_runtime")
    assert compatibility["contracts"] == {
        "experiment": "1.0",
        "handoff": "1.0",
        "report_facts": "2.0",
        "training": "2.0",
    }
    assert compatibility["source"]["repository"].endswith("/Qsaria")
    assert len(compatibility["source"]["commit"]) == 40
    assert set(compatibility["signatures"]) == {
        "source_contracts",
        "runtime",
        "bundle",
    }
    assert compatibility["scientific_pilot"]["status"] in {
        "pending",
        "passed",
        "failed",
    }


def test_shared_runtime_validator_passes_without_scientific_writes() -> None:
    result = _run_script("validate_shared_runtime.py")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "supported clients: codex_v1, claude_code_v1, claude_science_v1" in result.stdout
    assert "deterministic tool inventory: 46 synchronous + 7 durable-operation" in result.stdout
    assert "bootstrap persistence writes: none" in result.stdout


def test_preflight_passes_without_claude_cli() -> None:
    result = _run_script("preflight.py")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "[PASS] shared runtime" in result.stdout
    assert "scientific pilot remains an explicit end-to-end validation" in result.stdout


def test_smoke_validator_skips_or_passes_without_installation() -> None:
    source = (SCRIPTS_ROOT / "smoke_claude_plugin.py").read_text(encoding="utf-8")
    assert 'plugin", "validate' in source
    assert 'plugin", "install' not in source
    assert 'marketplace", "add' not in source


def test_personal_marketplace_is_local_and_not_public() -> None:
    marketplace = json.loads(
        (REPO_ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    assert marketplace["name"] == "personal"
    entries = [entry for entry in marketplace["plugins"] if entry["name"] == "qsaria-claude-code"]
    assert len(entries) == 1
    assert entries[0]["source"] == "./plugins/qsaria-claude-code"
    assert entries[0]["strict"] is True
    serialized = json.dumps(marketplace).lower()
    assert "claude-community" not in serialized
    assert "claude-plugins-official" not in serialized


def test_project_enables_only_the_personal_qsaria_plugin() -> None:
    settings = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert settings == {"enabledPlugins": {"qsaria-claude-code@personal": True}}


def test_plugin_activates_the_main_coordinator() -> None:
    manifest = json.loads(
        (PLUGIN_ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    settings = json.loads((PLUGIN_ROOT / "settings.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "qsaria-claude-code"
    assert manifest["displayName"] == "Qsaria for Claude Code"
    assert manifest["version"] == "0.4.0"
    assert "agents" not in manifest
    assert settings == {"agent": "qsaria-coordinator"}


def test_mcp_launcher_is_claude_native_and_deterministic() -> None:
    payload = json.loads((PLUGIN_ROOT / ".mcp.json").read_text(encoding="utf-8"))
    server = payload["mcpServers"]["qsaria"]
    assert server["type"] == "stdio"
    assert server["command"] == "uv"
    assert server["args"][:5] == [
        "--directory",
        "${CLAUDE_PROJECT_DIR}",
        "run",
        "--no-sync",
        "cscopilot-mcp",
    ]
    assert server["args"][server["args"].index("--profile") + 1] == "qsaria"
    assert server["args"][server["args"].index("--llm-policy") + 1] == "disabled"
    assert server["timeout"] == 3_600_000
    assert server["env"] == {
        "QSARIA_MODEL_CATALOG_PATH": "data/model_assets/catalog/qsaria_model_catalog.json"
    }
    assert "tool_timeout_sec" not in server
    assert "startup_timeout_sec" not in server
    assert "required" not in server


def test_claude_and_codex_share_exact_role_tool_contracts() -> None:
    claude = json.loads(
        (PLUGIN_ROOT / "contracts" / "agent-tools.json").read_text(encoding="utf-8")
    )
    codex = json.loads(
        (REPO_ROOT / "plugins" / "qsaria-codex" / "contracts" / "agent-tools.json").read_text(
            encoding="utf-8"
        )
    )
    assert claude == codex
    all_tools = {tool for tools in claude["roles"].values() for tool in tools}
    all_tools.update(
        {
            "qsaria_create_experiment",
            "qsaria_list_experiments",
            "qsaria_complete_experiment",
        }
    )
    assert len(all_tools) == 46


def test_specialists_are_sonnet_medium_non_recursive_and_mcp_only() -> None:
    contract = json.loads(
        (PLUGIN_ROOT / "contracts" / "agent-tools.json").read_text(encoding="utf-8")
    )
    forbidden = {"Agent", "SendMessage", "Read", "Bash", "Write", "Edit", "Skill"}
    for role, (agent_name, skill_name) in ROLE_AGENTS.items():
        metadata, body = _frontmatter(PLUGIN_ROOT / "agents" / f"{agent_name}.md")
        assert metadata["model"] == "sonnet"
        assert metadata["effort"] == "medium"
        assert metadata["maxTurns"] == 30
        assert metadata["background"] is False
        assert metadata["skills"] == [f"qsaria-claude-code:{skill_name}"]
        expected = {TOOL_PREFIX + tool for tool in contract["roles"][role]}
        assert set(metadata["tools"]) == expected
        assert not forbidden.intersection(metadata["tools"])
        assert "mcpServers" not in metadata
        assert "permissionMode" not in metadata
        assert "memory" not in metadata
        assert "exactly one fresh public handoff" in " ".join(body.split())


def test_coordinator_is_main_profile_not_a_sixth_specialist() -> None:
    metadata, body = _frontmatter(PLUGIN_ROOT / "agents" / "qsaria-coordinator.md")
    assert metadata["model"] == "inherit"
    assert "effort" not in metadata
    assert metadata["skills"] == ["qsaria-claude-code:qsaria-coordinate"]
    agent_tools = [tool for tool in metadata["tools"] if tool.startswith("Agent(")]
    assert len(agent_tools) == 1
    for agent_name, _ in ROLE_AGENTS.values():
        assert f"qsaria-claude-code:{agent_name}" in agent_tools[0]
    for forbidden in ("Read", "Bash", "Write", "Edit", "Skill", "SendMessage"):
        assert forbidden not in metadata["tools"]
    assert "not a sixth specialist" in body
    assert "Never patch code" in body


def test_skills_are_self_contained_preload_only_contracts() -> None:
    expected = {
        "qsaria-coordinate",
        "qsaria-curate",
        "qsaria-train",
        "qsaria-registry",
        "qsaria-infer",
        "qsaria-report",
    }
    actual = {path.parent.name for path in (PLUGIN_ROOT / "skills").glob("*/SKILL.md")}
    assert actual == expected
    for name in expected:
        metadata, body = _frontmatter(PLUGIN_ROOT / "skills" / name / "SKILL.md")
        assert metadata["name"] == name
        assert metadata["user-invocable"] is False
        assert metadata.get("disable-model-invocation") is not True
        assert "](references/" not in body
        assert "../../contracts/" not in body
        assert len(body.splitlines()) <= 220


def test_public_handoff_rule_from_latest_codex_fix_is_preserved() -> None:
    invariants = (PLUGIN_ROOT / "contracts" / "scientific-invariants.md").read_text(
        encoding="utf-8"
    )
    assert "source material, not the public envelope" in invariants
    assert "Build a fresh envelope" in invariants
    for name in ("qsaria-curate", "qsaria-train"):
        skill = (PLUGIN_ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
        assert "fresh envelope" in skill
        assert "never" in skill.lower() and "directly" in skill.lower()


def test_scientific_profile_cannot_enter_code_maintenance() -> None:
    coordinator = (PLUGIN_ROOT / "skills" / "qsaria-coordinate" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "never authorizes source edits" in coordinator
    assert "one coherent manual recovery mission" in coordinator
    assert "Never publish an Issue without explicit user approval" in " ".join(coordinator.split())
    assert "Claude analysis" in coordinator


def test_no_hooks_or_memory_are_shipped_inside_the_plugin() -> None:
    assert not (PLUGIN_ROOT / "hooks").exists()
    assert not (PLUGIN_ROOT / "CLAUDE.md").exists()
    assert not (PLUGIN_ROOT / "monitors").exists()


def test_claude_issue_form_requires_sanitized_recovery_evidence() -> None:
    issue = (REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "qsaria_claude_code_bug.yml").read_text(
        encoding="utf-8"
    )
    assert "[Qsaria Claude Code]" in issue
    assert "Manual recovery attempted" in issue
    assert "absolute home paths" in issue
    assert "private scientific results" in issue


def test_readme_documents_personal_project_scoped_installation() -> None:
    readme = (PLUGIN_ROOT / "README.md").read_text(encoding="utf-8")
    assert "qsaria-claude-code@personal" in readme
    assert "--scope project" in readme
    assert "not submitted to an Anthropic public marketplace" in " ".join(readme.split())
    assert "uv sync --extra mcp --extra prediction" in readme
