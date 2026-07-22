"""Static acceptance tests for the Qsaria Claude Science integration."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPO_ROOT / "plugins" / "qsaria-claude-science"
SCRIPTS_ROOT = BUNDLE_ROOT / "scripts"


def _run(name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / name), *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_science_bundle_validator_passes() -> None:
    result = _run("validate_science_bundle.py")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "46 synchronous + 7 durable-operation tools" in result.stdout


def test_science_compatibility_signatures_are_current() -> None:
    result = _run("sync_qsaria_science.py", "--check")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    compatibility = json.loads(
        (BUNDLE_ROOT / "contracts" / "compatibility.json").read_text(encoding="utf-8")
    )
    assert compatibility["client_contract"] == "claude_science_v1"
    assert compatibility["supported_clients"] == [
        "codex_v1",
        "claude_code_v1",
        "claude_science_v1",
    ]
    assert compatibility["minimum_claude_science_version"] == "0.1.21"
    assert compatibility["operation_contract"] == {
        "schema_version": "1.0",
        "synchronous_tools": 46,
        "durable_operation_tools": 7,
        "poll_interval_seconds": 20,
        "cancellation": False,
    }


def test_science_is_not_a_claude_code_plugin_or_marketplace_entry() -> None:
    assert not (BUNDLE_ROOT / ".claude-plugin").exists()
    marketplace = json.loads(
        (REPO_ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    assert "qsaria-claude-science" not in json.dumps(marketplace)


def test_launcher_uses_repo_python_and_shared_catalog() -> None:
    launcher = (SCRIPTS_ROOT / "launch-mcp.sh").read_text(encoding="utf-8")
    assert "/.venv/bin/python" in launcher
    assert "QSARIA_MODEL_CATALOG_PATH" in launcher
    assert "data/model_assets/catalog/qsaria_model_catalog.json" in launcher
    assert "QSARIA_SCIENCE_ARTIFACT_ROOT" in launcher
    assert "USE_S3=false" in launcher
    assert "--profile qsaria" in launcher
    assert "--llm-policy disabled" in launcher


def test_profile_contract_has_five_specialists_and_exact_detached_operations() -> None:
    contract = json.loads(
        (BUNDLE_ROOT / "contracts" / "agent-tools.json").read_text(encoding="utf-8")
    )
    assert contract["coordinator_profile"] == "QSARIA_COORDINATOR"
    assert set(contract["roles"]) == {
        "QSARIA_CURATION",
        "QSARIA_TRAINING",
        "QSARIA_REGISTRY",
        "QSARIA_INFERENCE",
        "QSARIA_REPORT",
    }
    assert contract["roles"]["QSARIA_REPORT"]["start_tool"] is None
    assert contract["roles"]["QSARIA_REPORT"]["detached_operations"] == []


def test_detached_probe_completes_outside_parent_call(tmp_path) -> None:
    started = _run("durability_probe.py", "start", "--root", str(tmp_path), "--seconds", "0.2")
    assert started.returncode == 0, started.stderr
    payload = json.loads(started.stdout)
    deadline = time.monotonic() + 5
    state_path = Path(payload["state"])
    while time.monotonic() < deadline:
        if state_path.is_file() and json.loads(state_path.read_text()).get("status") == "completed":
            break
        time.sleep(0.05)
    checked = _run("durability_probe.py", "check", "--state", str(state_path))
    assert checked.returncode == 0, checked.stdout


def test_preflight_is_read_only_and_reports_exact_connector() -> None:
    before = {path: path.stat().st_mtime_ns for path in (REPO_ROOT / ".files").glob("*")}
    result = _run("preflight.py")
    assert result.returncode in {0, 1}, result.stderr
    report = json.loads(result.stdout)
    assert report["connector"]["name"] == "qsaria"
    assert report["connector"]["command"] == "/bin/zsh"
    assert report["status"] in {"ok", "needs_configuration"}
    assert report["minimum_claude_science_version"] == "0.1.21"
    after = {path: path.stat().st_mtime_ns for path in (REPO_ROOT / ".files").glob("*")}
    assert after == before
