"""Tests for the MCP command-line entry point."""

from __future__ import annotations

import pytest


def test_main_loads_dotenv_before_bootstrap(monkeypatch):
    from cs_copilot.mcp import __main__ as entrypoint
    from cs_copilot.mcp import server as server_module

    calls: list[str] = []

    class FakeServer:
        def run(self, transport: str, **kwargs) -> None:
            calls.append(f"run:{transport}")

    def fake_build_server(*args, **kwargs):
        calls.append("build_server")
        return FakeServer()

    monkeypatch.setattr(entrypoint, "load_dotenv", lambda: calls.append("load_dotenv"))
    monkeypatch.setattr(entrypoint, "require_mcp", lambda: calls.append("require_mcp"))
    monkeypatch.setattr(
        entrypoint,
        "configure_logging",
        lambda log_level: calls.append(f"configure_logging:{log_level}"),
    )
    monkeypatch.setattr(
        entrypoint,
        "apply_session_id",
        lambda session_id: calls.append(f"apply_session_id:{session_id}"),
    )
    monkeypatch.setattr(entrypoint, "bootstrap", lambda config: calls.append("bootstrap") or {})
    monkeypatch.setattr(server_module, "build_server", fake_build_server)

    entrypoint.main(
        [
            "--session-id",
            "dotenv-test",
            "--log-level",
            "warning",
            "--no-tools",
            "--no-prompts",
            "--no-resources",
            "--no-chatgpt-compat",
        ]
    )

    assert calls[:5] == [
        "load_dotenv",
        "require_mcp",
        "configure_logging:warning",
        "apply_session_id:dotenv-test",
        "bootstrap",
    ]
    assert calls[-2:] == ["build_server", "run:stdio"]


def test_qsaria_profile_forces_disabled_llm_and_isolated_surface(monkeypatch):
    from cs_copilot.mcp import __main__ as entrypoint
    from cs_copilot.mcp import server as server_module

    observed: dict[str, object] = {}

    class FakeServer:
        def run(self, transport: str, **kwargs) -> None:
            observed["transport"] = transport

    def fake_bootstrap(config):
        observed["config"] = config
        return object()

    def fake_build_server(*args, **kwargs):
        observed["server_kwargs"] = kwargs
        return FakeServer()

    monkeypatch.setattr(entrypoint, "load_dotenv", lambda: None)
    monkeypatch.setattr(entrypoint, "require_mcp", lambda: None)
    monkeypatch.setattr(entrypoint, "configure_logging", lambda _level: None)
    monkeypatch.setattr(entrypoint, "apply_session_id", lambda _session_id: None)
    monkeypatch.setattr(entrypoint, "bootstrap", fake_bootstrap)
    monkeypatch.setattr(server_module, "build_server", fake_build_server)

    entrypoint.main(["--profile", "qsaria", "--llm-policy", "agno-model"])

    config = observed["config"]
    assert config.llm_policy == "disabled"
    kwargs = observed["server_kwargs"]
    assert kwargs["profile"] == "qsaria"
    assert kwargs["include_chatgpt_compat"] is False
    assert kwargs["include_prompts"] is False
    assert kwargs["include_resources"] is False
    assert kwargs["enable_agno_team_tool"] is False
    assert observed["transport"] == "stdio"


def test_qsaria_profile_rejects_private_agno_team_tool():
    from cs_copilot.mcp.__main__ import _parse_args

    with pytest.raises(SystemExit):
        _parse_args(["--profile", "qsaria", "--enable-agno-team-tool"])
