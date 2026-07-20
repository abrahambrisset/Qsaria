"""Static acceptance tests for the Qsaria Codex distribution bundle."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = REPO_ROOT / "plugins" / "qsaria-codex"
SCRIPTS_ROOT = PLUGIN_ROOT / "scripts"


def _run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / name), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_bundle_validator_passes() -> None:
    result = _run_script("validate_bundle.py")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_compatibility_signatures_are_current() -> None:
    result = _run_script("sync_qsaria_codex.py", "--check")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_mcp_launcher_is_isolated_and_bounded() -> None:
    payload = json.loads((PLUGIN_ROOT / ".mcp.json").read_text(encoding="utf-8"))
    server = payload["mcpServers"]["qsaria"]
    args = server["args"]
    assert server["command"] == "uv"
    assert args[:3] == ["run", "--no-sync", "cscopilot-mcp"]
    assert args[args.index("--profile") + 1] == "qsaria"
    assert args[args.index("--llm-policy") + 1] == "disabled"
    assert {"--no-chatgpt-compat", "--no-prompts", "--no-resources"} <= set(args)
    assert server["tool_timeout_sec"] == 3600
    assert "env" not in server
    serialized = json.dumps(server)
    assert "jobs.py" not in serialized
    assert "worker.py" not in serialized


def test_profile_contract_contains_46_tools() -> None:
    contract = json.loads(
        (PLUGIN_ROOT / "contracts" / "agent-tools.json").read_text(encoding="utf-8")
    )
    tools = {tool for role in contract["roles"].values() for tool in role}
    tools.update(
        {
            "qsaria_create_experiment",
            "qsaria_list_experiments",
            "qsaria_complete_experiment",
        }
    )
    assert len(tools) == 46


def test_report_handoff_schema_matches_single_mission_runtime_policy() -> None:
    schema = json.loads(
        (PLUGIN_ROOT / "contracts" / "handoff.schema.json").read_text(encoding="utf-8")
    )
    report_status_rule = schema["allOf"][0]["then"]["properties"]["status"]["enum"]
    assert report_status_rule == ["completed", "partial", "terminal_failure"]
    report_artifact_rule = schema["allOf"][1]["then"]["properties"]["artifact_ids"]
    assert report_artifact_rule == {"minItems": 1, "maxItems": 1}
    assert schema["properties"]["execution_mode"]["enum"] == [
        "project_agent",
        "coordinator_manual_tools",
    ]


def test_coordinator_recovery_never_implies_code_or_github_authority() -> None:
    skill_root = PLUGIN_ROOT / "skills" / "qsaria-coordinate"
    coordinator = (skill_root / "SKILL.md").read_text(encoding="utf-8")
    recovery = (skill_root / "references" / "manual-recovery-and-issues.md").read_text(
        encoding="utf-8"
    )
    invariants = (PLUGIN_ROOT / "contracts" / "scientific-invariants.md").read_text(
        encoding="utf-8"
    )

    assert "Never edit source" in coordinator
    assert "coordinator_manual_tools" in coordinator
    assert "Never publish an issue without the user's explicit approval" in coordinator
    assert "one coherent manual recovery mission" in recovery
    assert "Do not use shell writes, `apply_patch`" in recovery
    assert "An approval to retry, continue, solve, configure" in recovery
    assert "GitHub writes" in invariants


def test_qsaria_github_issue_form_captures_recovery_and_privacy() -> None:
    issue_form = (REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "qsaria_codex_bug.yml").read_text(
        encoding="utf-8"
    )
    assert "[Qsaria Codex]" in issue_form
    assert "Manual recovery attempted" in issue_form
    assert "Fresh Codex task reproduced" in issue_form
    assert "absolute home paths" in issue_form


def test_agents_are_read_only_non_recursive_and_contract_driven() -> None:
    contract = json.loads(
        (PLUGIN_ROOT / "contracts" / "agent-tools.json").read_text(encoding="utf-8")
    )
    project = tomllib.loads(
        (PLUGIN_ROOT / "assets" / "project-config.toml").read_text(encoding="utf-8")
    )
    assert project["agents"]["max_depth"] == 1
    assert project["agents"]["max_threads"] == 5

    for role, expected_tools in contract["roles"].items():
        template = tomllib.loads(
            (PLUGIN_ROOT / "assets" / "agents" / f"{role}.toml").read_text(encoding="utf-8")
        )
        assert template["model"] == "gpt-5.6-terra"
        assert template["model_reasoning_effort"] == "medium"
        assert template["sandbox_mode"] == "read-only"
        assert "Never spawn" in template["developer_instructions"]
        assert template["developer_instructions"].count("qsaria_record_handoff") == 1
        assert 'schema_version = "1.0"' in template["developer_instructions"]
        assert 'execution_mode = "project_agent"' in template["developer_instructions"]
        assert "toolkit-returned handoff" in template["developer_instructions"]
        server = template["plugins"]["qsaria-codex"]["mcp_servers"]["qsaria"]
        assert server["enabled_tools"] == expected_tools
        assert expected_tools.count("qsaria_record_handoff") == 1


def test_active_project_agents_match_distribution_templates() -> None:
    active_root = REPO_ROOT / ".codex" / "agents"
    template_root = PLUGIN_ROOT / "assets" / "agents"
    template_names = {path.name for path in template_root.glob("qsaria_*.toml")}
    active_names = {path.name for path in active_root.glob("qsaria_*.toml")}
    assert active_names == template_names
    for name in template_names:
        assert (active_root / name).read_bytes() == (template_root / name).read_bytes()

    active_config = tomllib.loads(
        (REPO_ROOT / ".codex" / "config.toml").read_text(encoding="utf-8")
    )
    template_config = tomllib.loads(
        (PLUGIN_ROOT / "assets" / "project-config.toml").read_text(encoding="utf-8")
    )
    assert active_config["agents"] == template_config["agents"]


def test_personal_marketplace_points_to_qsaria_plugin() -> None:
    marketplace = json.loads(
        (REPO_ROOT / ".agents" / "plugins" / "marketplace.json").read_text(encoding="utf-8")
    )
    assert marketplace["name"] == "personal"
    assert marketplace["interface"]["displayName"] == "Personal"
    entries = [item for item in marketplace["plugins"] if item.get("name") == "qsaria-codex"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["source"] == {
        "source": "local",
        "path": "./plugins/qsaria-codex",
    }
    assert entry["policy"]["installation"] == "AVAILABLE"
    assert entry["policy"]["authentication"] == "ON_INSTALL"
    assert entry["category"] == "Developer Tools"


def test_role_skills_are_not_implicitly_selected() -> None:
    for skill_root in (PLUGIN_ROOT / "skills").iterdir():
        if not skill_root.is_dir():
            continue
        text = (skill_root / "agents" / "openai.yaml").read_text(encoding="utf-8")
        expected = "true" if skill_root.name == "qsaria-coordinate" else "false"
        assert f"allow_implicit_invocation: {expected}" in text


def test_standard_lightgbm_contract_and_registry_routing_are_explicit() -> None:
    training_policy = (
        PLUGIN_ROOT / "skills" / "qsaria-train" / "references" / "training-policy.md"
    ).read_text(encoding="utf-8")
    coordinator_routing = (
        PLUGIN_ROOT / "skills" / "qsaria-coordinate" / "references" / "routing.md"
    ).read_text(encoding="utf-8")
    registry_policy = (
        PLUGIN_ROOT / "skills" / "qsaria-registry" / "references" / "model-lifecycle.md"
    ).read_text(encoding="utf-8")

    for text in (training_policy, coordinator_routing):
        assert 'representation_name="rdkit_all"' in text or "representation `rdkit_all`" in text
        assert 'validation_protocol="standard_qsar"' in text or "protocol `standard_qsar`" in text
        assert "50 requested trials" in text
        assert "outlier analysis" in text
        assert "simple prompt" in text
    assert "`baseline` or `simplified`, never `standard`" in training_policy
    assert "`candidate_manifest_path`" in registry_policy
    assert "`recommended_registry_payload`" in registry_policy
    assert "Never call the batch persistence tool" in registry_policy
    assert "persistence.status=pending" in registry_policy
    assert "persistence_policy=session_only" in registry_policy


def test_report_skill_receives_all_table_material_but_keeps_editorial_choice() -> None:
    report_skill = (PLUGIN_ROOT / "skills" / "qsaria-report" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    report_contract = (
        PLUGIN_ROOT / "skills" / "qsaria-report" / "references" / "report-contract.md"
    ).read_text(encoding="utf-8")

    assert "verified editorial material, not mandatory output blocks" in report_skill
    assert "Select zero to three tables" in report_skill
    assert "omit redundant rows or columns" in report_contract
    assert "must come unchanged from the structured table" in report_contract


def test_standard_lightgbm_contract_matches_current_qsaria_defaults() -> None:
    from cs_copilot.tools.prediction.hyperparameter_tuning import normalize_tuning_config
    from cs_copilot.tools.prediction.outlier_analysis import normalize_outlier_analysis_config
    from cs_copilot.tools.prediction.tabular_representations import (
        default_tabular_representation_for_protocol,
    )

    tuning = normalize_tuning_config(
        None,
        backend_name="lightgbm",
        task_type="regression",
        eligible=True,
    )
    outliers, skip_reason = normalize_outlier_analysis_config(
        None,
        has_validation=True,
        target_count=1,
    )

    assert default_tabular_representation_for_protocol("standard_qsar") == "rdkit_all"
    assert tuning is not None
    assert tuning.enabled is True
    assert tuning.n_trials == 50
    assert outliers.enabled is True
    assert skip_reason is None


def test_cachebuster_replaces_suffix_without_changing_base_version() -> None:
    sys.path.insert(0, str(SCRIPTS_ROOT))
    try:
        from update_plugin_cachebuster import _sanitize_cachebuster, _with_cachebuster
    finally:
        sys.path.pop(0)

    assert _with_cachebuster("0.1.0", "local-1") == "0.1.0+codex.local-1"
    assert _with_cachebuster("0.1.0+codex.previous", "local-2") == "0.1.0+codex.local-2"
    assert _sanitize_cachebuster(" Local  2026/07/17 ") == "local-2026-07-17"


def test_preflight_distinguishes_mcp_from_training_readiness() -> None:
    sys.path.insert(0, str(SCRIPTS_ROOT))
    try:
        from preflight import _training_readiness
    finally:
        sys.path.pop(0)

    ready, detail = _training_readiness(
        {
            "chemprop": {
                "available": True,
                "importable": True,
                "package_version": "2.2.3",
            },
            "lightgbm": {
                "available": True,
                "importable": True,
                "package_version": "4.6.0",
            },
            "tabicl": {
                "available": True,
                "importable": True,
                "package_version": "2.1.1",
            },
        }
    )
    assert ready is True
    assert "lightgbm=4.6.0" in detail

    ready, detail = _training_readiness(
        {
            "chemprop": {
                "available": True,
                "importable": True,
                "package_version": "2.2.3",
            },
            "lightgbm": {
                "available": False,
                "importable": False,
                "package_version": None,
                "import_error": "ModuleNotFoundError: lightgbm",
            },
            "tabicl": {
                "available": True,
                "importable": True,
                "package_version": "2.1.1",
            },
        }
    )
    assert ready is False
    assert "lightgbm" in detail
    assert "uv sync --extra mcp --extra prediction" in detail


def test_readme_requires_training_extra_and_documents_mcp_only_mode() -> None:
    readme = (PLUGIN_ROOT / "README.md").read_text(encoding="utf-8")
    assert "uv sync --extra mcp --extra prediction" in readme
    assert "preflight.py --mcp-only" in readme


def test_compatibility_detects_source_provenance_drift() -> None:
    sys.path.insert(0, str(SCRIPTS_ROOT))
    try:
        from sync_qsaria_codex import compatibility_differences
    finally:
        sys.path.pop(0)

    expected = {
        "schema_version": "1.0",
        "plugin_version": "0.1.0",
        "source": {
            "commit": "abc",
            "tree": "tree-a",
            "signed_tree": "sha256:signed-a",
            "dirty": False,
        },
        "contracts": {"experiment": "1.0"},
        "signatures": {"bundle": "sha256:bundle-a"},
    }
    current = copy.deepcopy(expected)
    assert compatibility_differences(expected, current) == []

    current["source"]["dirty"] = True
    assert compatibility_differences(expected, current) == ["source"]
    current = copy.deepcopy(expected)
    current["source"]["commit"] = "def"
    assert compatibility_differences(expected, current) == ["source"]


def test_compatibility_covers_supporting_scientific_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(SCRIPTS_ROOT))
    try:
        from sync_qsaria_codex import (
            REPOSITORY_INTEGRATION_FILES,
            SUPPORTING_SCIENTIFIC_FILES,
            _signed_source_files,
            _tool_surface_signature,
        )
    finally:
        sys.path.pop(0)

    expected = {REPO_ROOT / relative for relative in SUPPORTING_SCIENTIFIC_FILES}
    assert expected <= set(_signed_source_files(REPO_ROOT))
    assert ".github/ISSUE_TEMPLATE/qsaria_codex_bug.yml" in REPOSITORY_INTEGRATION_FILES
    assert (REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "qsaria_codex_bug.yml").resolve() in set(
        _signed_source_files(REPO_ROOT)
    )

    baseline = _tool_surface_signature(REPO_ROOT)
    target = REPO_ROOT / SUPPORTING_SCIENTIFIC_FILES[0]
    original_read_bytes = Path.read_bytes

    def _read_bytes_with_drift(path: Path) -> bytes:
        content = original_read_bytes(path)
        if path == target:
            return content + b"\n# compatibility drift\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", _read_bytes_with_drift)
    assert _tool_surface_signature(REPO_ROOT) != baseline


def test_codex_plugin_smoke_uses_only_a_temporary_home() -> None:
    result = _run_script("smoke_codex_plugin.py", "--require-codex")
    if "Codex CLI unavailable" in result.stderr:
        return
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    payload = json.loads(result.stdout)
    assert payload == {
        "ok": True,
        "marketplace": "personal",
        "plugin": "qsaria-codex",
        "isolated": True,
    }
