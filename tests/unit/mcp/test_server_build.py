"""End-to-end smoke test for FastMCP server assembly.

Builds the server in-process (no stdio) and asserts that every registered
tool, prompt, and the resource manifest are present.
"""

from __future__ import annotations

import asyncio
import os

import pytest

mcp = pytest.importorskip("mcp")

from cs_copilot.mcp.session import (  # noqa: E402  (import-after-skip)
    BootstrapConfig,
    apply_session_id,
    bootstrap,
)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    previous_workdir = os.getcwd()
    work = tmp_path_factory.mktemp("mcp-server")
    os.chdir(work)
    apply_session_id("mcp-server-test")
    ctx = bootstrap(BootstrapConfig(session_id="mcp-server-test", workflow_slug="smoke"))
    from cs_copilot.mcp.server import build_server

    try:
        yield build_server(ctx)
    finally:
        os.chdir(previous_workdir)


@pytest.fixture(scope="module")
def qsaria_server(tmp_path_factory):
    from cs_copilot.mcp.context import MCPAgentContext
    from cs_copilot.mcp.server import build_server

    previous_workdir = os.getcwd()
    work = tmp_path_factory.mktemp("qsaria-mcp-server")
    os.chdir(work)
    try:
        yield build_server(
            MCPAgentContext(llm_policy="disabled"),
            profile="qsaria",
            include_chatgpt_compat=True,
            include_prompts=True,
            include_resources=True,
            enable_agno_team_tool=True,
        )
    finally:
        os.chdir(previous_workdir)


def test_server_instructions_are_chatgpt_orchestration_contract(server):
    from cs_copilot.mcp.server import SERVER_INSTRUCTIONS

    assert server.instructions == SERVER_INSTRUCTIONS
    assert server._mcp_server.instructions == SERVER_INSTRUCTIONS
    assert len(SERVER_INSTRUCTIONS) <= 512
    assert "external MCP client is the reasoning layer" in SERVER_INSTRUCTIONS
    assert "Do not invoke the Agno team" in SERVER_INSTRUCTIONS
    assert "mcp_bootstrap" in SERVER_INSTRUCTIONS
    assert "cs_copilot_mcp_workflow" in SERVER_INSTRUCTIONS
    assert "cs_copilot_workflow" in SERVER_INSTRUCTIONS
    assert "chembl_prepare_retrieval" in SERVER_INSTRUCTIONS
    assert "chemspace_plan_analysis" in SERVER_INSTRUCTIONS
    assert "llm_*" in SERVER_INSTRUCTIONS
    assert "chembl_retrieval_judge" in SERVER_INSTRUCTIONS


def test_bootstrap_attaches_default_external_llm_broker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    apply_session_id("mcp-llm-policy-test")
    ctx = bootstrap(BootstrapConfig(session_id="mcp-llm-policy-test", workflow_slug="smoke"))

    assert ctx.llm_policy == "external"
    assert ctx.model is None
    assert ctx.llm is not None


def test_tools_registered(server):
    names = sorted(t.name for t in server._tool_manager.list_tools())
    assert "search" in names
    assert "fetch" in names
    assert "mcp_bootstrap" in names
    assert "chembl_prepare_retrieval" in names
    assert "chembl_fetch_compounds" in names
    assert "chemspace_plan_analysis" in names
    assert "gtm_optimization" in names
    assert "report_save_markdown" in names
    assert "skill_fetch" in names
    assert "workflow_fetch" in names
    assert "llm_list_pending_tasks" in names
    assert "llm_submit_task_result" in names
    assert "pandas_create_dataframe" in names
    assert "mol_validate_design_candidates" in names
    assert "peptide_validate_design_candidates" in names
    assert "synplanner_identify_input" in names
    assert "agno_team_run" not in names


def test_qsaria_profile_exposes_only_its_deterministic_surface(qsaria_server):
    from cs_copilot.mcp.tools_registry import all_specs

    names = {tool.name for tool in qsaria_server._tool_manager.list_tools()}
    expected = {spec.mcp_name for spec in all_specs(profile="qsaria")}

    assert len(expected) == 46
    assert names == expected
    assert "qsaria_bootstrap" in names
    assert "qsaria_report_save" in names
    assert "search" not in names
    assert "fetch" not in names
    assert "agno_team_run" not in names
    assert "mcp_bootstrap" not in names
    assert not any(spec.run_in_worker_process for spec in all_specs(profile="qsaria"))


def test_qsaria_profile_forces_prompts_resources_and_agno_off(qsaria_server):
    from cs_copilot.mcp.server import QSARIA_SERVER_INSTRUCTIONS

    assert qsaria_server.name == "qsaria"
    assert qsaria_server.instructions == QSARIA_SERVER_INSTRUCTIONS
    assert qsaria_server._prompt_manager.list_prompts() == []
    assert asyncio.run(qsaria_server.list_resources()) == []


def test_qsaria_annotations_do_not_label_state_indexing_as_read_only(qsaria_server):
    tools = {tool.name: tool for tool in qsaria_server._tool_manager.list_tools()}

    assert tools["qsaria_registry_list_catalog_models"].annotations.readOnlyHint is True
    assert tools["qsaria_get_experiment_state"].annotations.readOnlyHint is True
    for name in {
        "qsaria_list_artifacts",
        "qsaria_get_artifact",
        "qsaria_report_build_context",
        "qsaria_registry_list_registered_models",
        "qsaria_registry_summarize_model",
    }:
        assert tools[name].annotations.readOnlyHint is False, name


def test_qsaria_bootstrap_does_not_resume_or_list_experiments(qsaria_server):
    tools = {tool.name: tool for tool in qsaria_server._tool_manager.list_tools()}

    result = asyncio.run(tools["qsaria_bootstrap"].fn())

    assert result["profile"] == "qsaria"
    assert result["llm_policy"] == "disabled"
    assert result["auto_resume"] is False
    assert result["active_experiment_id"] is None
    assert result["experiments_listed"] is False


def test_qsaria_protocol_calls_preserve_structured_content(qsaria_server):
    content, structured = asyncio.run(
        qsaria_server._tool_manager.call_tool(
            "qsaria_bootstrap",
            {},
            convert_result=True,
        )
    )

    assert content
    assert structured["profile"] == "qsaria"
    assert structured["contracts"]["handoff"] == "1.0"


def test_qsaria_lifecycle_tools_share_one_durable_manager(qsaria_server):
    tools = {tool.name: tool for tool in qsaria_server._tool_manager.list_tools()}
    experiment_id = "exp_server_lifecycle_123"

    created = asyncio.run(
        tools["qsaria_create_experiment"].fn(
            user_request="Train a test model",
            report_language="en",
            metadata={"source": "server-smoke"},
            experiment_id=experiment_id,
        )
    )
    reopened = asyncio.run(tools["qsaria_open_experiment"].fn(experiment_id=experiment_id))
    environment = asyncio.run(tools["qsaria_training_describe_outlier_analysis"].fn())

    assert created["experiment_id"] == experiment_id
    assert reopened == created
    assert reopened["report_language"] == "en"
    assert environment


def test_qsaria_profile_rejects_any_attached_model():
    from cs_copilot.mcp.context import MCPAgentContext
    from cs_copilot.mcp.server import build_server

    with pytest.raises(ValueError, match="llm_policy='disabled'"):
        build_server(MCPAgentContext(llm_policy="external"), profile="qsaria")
    with pytest.raises(ValueError, match="no model"):
        build_server(
            MCPAgentContext(llm_policy="disabled", model=object()),
            profile="qsaria",
        )


def test_all_direct_parity_tools_registered(server):
    names = {tool.name for tool in server._tool_manager.list_tools()}

    expected = {
        "skill_list",
        "skill_search",
        "skill_fetch",
        "mcp_bootstrap",
        "workflow_list",
        "workflow_search",
        "workflow_fetch",
        "llm_create_task",
        "llm_list_pending_tasks",
        "llm_get_task",
        "llm_submit_task_result",
        "llm_cancel_task",
        "pandas_load_dataframe_from_session",
        "pandas_create_dataframe",
        "pandas_run_operation",
        "pandas_normalize_for_analysis",
        "mol_list_design_engines",
        "mol_design_molecules",
        "mol_generate_analogs",
        "mol_interpolate_molecules",
        "mol_validate_design_candidates",
        "mol_rank_design_candidates",
        "mol_register_design_candidates",
        "peptide_list_design_engines",
        "peptide_design_peptides",
        "peptide_generate_analogs",
        "peptide_design_interpolation",
        "peptide_validate_design_candidates",
        "peptide_rank_design_candidates",
        "peptide_load_design_candidates",
        "peptide_validate_model_loaded",
        "peptide_get_latent_dimension",
        "peptide_encode_peptides",
        "peptide_decode_latent",
        "peptide_sample_peptides",
        "peptide_interpolate_peptides",
        "peptide_reconstruct_sequence",
        "peptide_explore_latent_neighborhood",
        "peptide_get_model_info",
        "synplanner_identify_input",
        "synplanner_convert_name_to_smiles",
        "synplanner_plan_synthesis",
        "synplanner_describe_plan",
        "synplanner_get_route_visualizations",
        "chembl_prepare_retrieval",
        "chembl_create_external_judge_task",
        "chembl_submit_external_judge_result",
        "chemspace_plan_analysis",
    }

    assert expected.issubset(names)


def test_tool_annotations_for_chatgpt_approval(server):
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}

    for name, tool in tools.items():
        assert tool.annotations is not None, name
        assert tool.annotations.destructiveHint is False, name
        assert tool.annotations.openWorldHint is False, name

    assert tools["search"].annotations.readOnlyHint is True
    assert tools["fetch"].annotations.readOnlyHint is True
    assert tools["chembl_prepare_retrieval"].annotations.readOnlyHint is True
    assert tools["chemspace_plan_analysis"].annotations.readOnlyHint is True
    assert tools["chem_calculate_tanimoto_similarity"].annotations.readOnlyHint is True
    assert tools["chembl_describe_dataset"].annotations.readOnlyHint is True
    assert tools["gtm_get_density_summary"].annotations.readOnlyHint is True
    assert tools["session_resolve_candidate_set"].annotations.readOnlyHint is True
    assert tools["skill_fetch"].annotations.readOnlyHint is True
    assert tools["mcp_bootstrap"].annotations.readOnlyHint is True
    assert tools["workflow_fetch"].annotations.readOnlyHint is True
    assert tools["llm_list_pending_tasks"].annotations.readOnlyHint is True
    assert tools["llm_get_task"].annotations.readOnlyHint is True
    assert tools["mol_validate_design_candidates"].annotations.readOnlyHint is True
    assert tools["peptide_validate_design_candidates"].annotations.readOnlyHint is True
    assert tools["synplanner_identify_input"].annotations.readOnlyHint is True

    assert tools["chembl_fetch_compounds"].annotations.readOnlyHint is False
    assert tools["gtm_optimization"].annotations.readOnlyHint is False
    assert tools["gtm_sample_nodes"].annotations.readOnlyHint is False
    assert tools["session_select_session_object"].annotations.readOnlyHint is False
    assert tools["session_summarize_session_memory"].annotations.readOnlyHint is False
    assert tools["report_save_markdown"].annotations.readOnlyHint is False
    assert tools["pandas_create_dataframe"].annotations.readOnlyHint is False
    assert tools["llm_create_task"].annotations.readOnlyHint is False
    assert tools["llm_submit_task_result"].annotations.readOnlyHint is False
    assert tools["llm_cancel_task"].annotations.readOnlyHint is False
    assert tools["mol_design_molecules"].annotations.readOnlyHint is False
    assert tools["peptide_design_peptides"].annotations.readOnlyHint is False
    assert tools["synplanner_plan_synthesis"].annotations.readOnlyHint is False


def test_agno_team_tool_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    apply_session_id("mcp-agno-opt-in-test")
    ctx = bootstrap(BootstrapConfig(session_id="mcp-agno-opt-in-test", workflow_slug="smoke"))
    from cs_copilot.mcp.server import build_server

    server = build_server(ctx, enable_agno_team_tool=True)
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}

    assert "agno_team_run" in tools
    assert tools["agno_team_run"].annotations.readOnlyHint is False


def test_prompts_registered(server):
    names = sorted(p.name for p in server._prompt_manager.list_prompts())
    assert "cs_copilot_mcp_workflow" in names
    assert "cs_copilot_workflow" in names
    assert "chembl_retrieval_judge" in names


def test_resources_include_manifest(server):
    entries = asyncio.run(server.list_resources())
    uris = {str(e.uri) for e in entries}
    assert "cscopilot://session/manifest.json" in uris


def test_transport_security_adds_public_proxy_hosts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    apply_session_id("mcp-transport-security-test")
    ctx = bootstrap(
        BootstrapConfig(session_id="mcp-transport-security-test", workflow_slug="smoke")
    )
    from cs_copilot.mcp.server import build_server

    server = build_server(
        ctx,
        allowed_hosts=["mcp.example.com"],
        allowed_origins=["https://chatgpt.com"],
    )

    security = server.settings.transport_security
    assert security.enable_dns_rebinding_protection is True
    assert "127.0.0.1:*" in security.allowed_hosts
    assert "localhost:*" in security.allowed_hosts
    assert "mcp.example.com" in security.allowed_hosts
    assert "http://127.0.0.1:*" in security.allowed_origins
    assert "https://chatgpt.com" in security.allowed_origins


def test_transport_security_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    apply_session_id("mcp-transport-security-disabled-test")
    ctx = bootstrap(
        BootstrapConfig(
            session_id="mcp-transport-security-disabled-test",
            workflow_slug="smoke",
        )
    )
    from cs_copilot.mcp.server import build_server

    server = build_server(ctx, disable_dns_rebinding_protection=True)

    security = server.settings.transport_security
    assert security.enable_dns_rebinding_protection is False


def test_transport_security_rejects_unlisted_host(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    apply_session_id("mcp-transport-security-http-test")
    ctx = bootstrap(
        BootstrapConfig(session_id="mcp-transport-security-http-test", workflow_slug="smoke")
    )
    from cs_copilot.mcp.server import build_server

    server = build_server(ctx)

    async def request_with_public_host():
        import httpx

        app = server.streamable_http_app()
        async with server.session_manager.run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.post(
                    "/mcp",
                    json={},
                    headers={"Host": "mcp.example.com"},
                )

    response = asyncio.run(request_with_public_host())

    assert response.status_code == 421


def test_transport_security_allows_configured_host(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    apply_session_id("mcp-transport-security-http-allow-test")
    ctx = bootstrap(
        BootstrapConfig(
            session_id="mcp-transport-security-http-allow-test",
            workflow_slug="smoke",
        )
    )
    from cs_copilot.mcp.server import build_server

    server = build_server(ctx, allowed_hosts=["mcp.example.com"])

    async def request_with_public_host():
        import httpx

        app = server.streamable_http_app()
        async with server.session_manager.run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.post(
                    "/mcp",
                    json={},
                    headers={"Host": "mcp.example.com"},
                )

    response = asyncio.run(request_with_public_host())

    assert response.status_code != 421


def test_bearer_auth_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    apply_session_id("mcp-auth-test")
    ctx = bootstrap(BootstrapConfig(session_id="mcp-auth-test", workflow_slug="smoke"))
    from cs_copilot.mcp.server import build_server

    protected = "https://mcp.example.com/mcp"
    server = build_server(
        ctx,
        auth_token="secret",
        auth_token_client_id="client-1",
        auth_token_scopes=["mcp:read"],
        auth_issuer_url=protected,
        auth_resource_url=protected,
    )

    assert str(server.settings.auth.issuer_url) == protected
    assert str(server.settings.auth.resource_server_url) == protected
    assert server.settings.auth.required_scopes == ["mcp:read"]
    token = asyncio.run(server._token_verifier.verify_token("secret"))
    assert token.client_id == "client-1"


def test_bearer_auth_rejects_missing_header(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    apply_session_id("mcp-auth-http-test")
    ctx = bootstrap(BootstrapConfig(session_id="mcp-auth-http-test", workflow_slug="smoke"))
    from cs_copilot.mcp.server import build_server

    server = build_server(
        ctx,
        auth_token="secret",
        auth_issuer_url="https://mcp.example.com/mcp",
        auth_resource_url="https://mcp.example.com/mcp",
    )

    async def request_without_auth():
        import httpx

        transport = httpx.ASGITransport(app=server.streamable_http_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/mcp", json={})

    response = asyncio.run(request_without_auth())

    assert response.status_code == 401
    assert response.json()["error"] == "invalid_token"
    assert response.headers["www-authenticate"].startswith("Bearer")
