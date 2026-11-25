"""Exercise Chainlit's real team and tool dispatch against a scripted local model transport."""

import ast
import csv
import json
import os
from hashlib import sha256
from itertools import count
from math import isclose
from pathlib import Path
from statistics import mean
from uuid import uuid4

import httpx
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat

root = Path(__file__).resolve().parents[1]
directory = root / ".files/publication/chainlit-roundtrip" / uuid4().hex
directory.mkdir(parents=True, exist_ok=True)
os.environ["MODEL_PROVIDER"] = "ollama"
os.environ["MODEL_ID"] = "offline-qsaria-fixture"
os.environ["CS_COPILOT_AGENT_TEAM"] = "qsar"

import chainlit_app  # noqa: E402

source = root / "benchmarks/fixtures/qsaria-v1-synthetic.csv"
curated = directory / "curated.csv"
model_id = "offline-qsaria-fixture"
sequence = count(1)
observed = {}
member_ids = {}
workflow_roles = ("curation", "training", "report")
training_request = {
    "target_columns": ["clogp"],
    "task_type": "regression",
    "representation": {"kind": "generated", "name": "morgan_only"},
    "backend": {"device": "cpu", "n_estimators": 12, "num_leaves": 6, "min_child_samples": 2},
    "validation": {"kind": "standard_qsar", "seed": 42},
    "compute": {"profile": "local_light"},
    "outlier_analysis": {"enabled": False},
    "applicability_domain": {"methods": ["bounding_box"]},
}


def parse_tool(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def handler(request):
    data = json.loads(request.content)
    assert not data.get("stream"), "round-trip fixture uses non-streaming Team.run"
    tools = {tool["function"]["name"] for tool in data.get("tools") or []}
    history = [msg for msg in data["messages"] if msg.get("role") == "tool"]
    arguments = None
    if "delegate_task_to_member" in tools:
        role_index = len(history)
        if role_index < len(workflow_roles):
            role = workflow_roles[role_index]
            arguments = (
                "delegate_task_to_member",
                {
                    "member_id": member_ids[role],
                    "task_description": f"Process the {role} phase of the local fixture.",
                    "expected_output": "Verified tool evidence.",
                },
            )
        else:
            content = observed["report"]
    elif "curate_qsar_dataset" in tools:
        if history:
            value = parse_tool(history[-1]["content"])
            observed["curation"] = value["rows_out"]
            content = f"Curated rows: {observed['curation']}"
        else:
            arguments = (
                "curate_qsar_dataset",
                {
                    "dataset_path": str(source),
                    "task_type": "regression",
                    "smiles_column": "smiles",
                    "target_columns": ["clogp"],
                    "output_csv": str(curated),
                },
            )
    elif "train_lightgbm_model" in tools:
        if history:
            value = parse_tool(history[-1]["content"])
            split = value["split_results"][0]
            observed["training"] = {
                "mae": split["metrics"]["test"]["mae"],
                "predictions": split["test_predictions_path"],
                "tuning": value["requested_contract"]["tuning"]["enabled"],
            }
            content = json.dumps(observed["training"])
        else:
            arguments = (
                "train_lightgbm_model",
                {"train_csv": str(curated), "request": training_request},
            )
    else:
        observed["report"] = (
            f"{observed['curation']} molecules retained. Test MAE {observed['training']['mae']:.6f}."
        )
        content = observed["report"]
    message = {"role": "assistant", "content": content if arguments is None else None}
    if arguments:
        name, payload = arguments
        message["tool_calls"] = [
            {
                "id": f"call_{next(sequence)}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(payload)},
            }
        ]
    return httpx.Response(
        200,
        json={
            "id": f"chatcmpl-{next(sequence)}",
            "object": "chat.completion",
            "created": 1,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if arguments else "stop",
                    "message": message,
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


with httpx.Client(transport=httpx.MockTransport(handler)) as http:
    chainlit_app.model = OpenAIChat(
        id=model_id,
        api_key="local-fixture",
        http_client=http,
        max_retries=0,
        base_url="http://qsaria-fixture.invalid/v1",
    )
    team = chainlit_app._create_session_agent("qsar")
    team.db = SqliteDb(db_file=str(directory / "memory.db"))
    team.store_member_responses = True
    for role in workflow_roles:
        member = next(member for member in team.members if role in member.name)
        member.set_id()
        member_ids[role] = member.id
    response = team.run(
        "Curate the fixture, train the requested LightGBM model and report the test result.",
        stream=False,
        session_id=f"qsaria-publication-roundtrip-{uuid4().hex}",
    )
assert observed["curation"] == 40
assert observed["training"]["tuning"] is False
assert [tool.tool_name for tool in response.tools] == ["delegate_task_to_member"] * len(
    workflow_roles
)
assert [tool.tool_args["member_id"] for tool in response.tools] == [
    member_ids[role] for role in workflow_roles
]
assert all(not tool.tool_call_error for tool in response.tools)
assert [member.agent_id for member in response.member_responses] == [
    member_ids[role] for role in workflow_roles
]
assert [
    [tool.tool_name for tool in member.tools or []] for member in response.member_responses
] == [["curate_qsar_dataset"], ["train_lightgbm_model"], []]
curation, training, report = response.member_responses
assert not curation.tools[0].tool_call_error
assert not training.tools[0].tool_call_error
assert training.tools[0].tool_args["train_csv"] == curation.tools[0].tool_args["output_csv"]

predictions = Path(observed["training"]["predictions"])
with predictions.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 4
mae = mean(abs(float(row["y_true"]) - float(row["y_pred"])) for row in rows)
assert isclose(mae, observed["training"]["mae"], rel_tol=1e-10, abs_tol=1e-10)
assert response.content == report.content == f"40 molecules retained. Test MAE {mae:.6f}."
observed["training"]["predictions_sha256"] = sha256(predictions.read_bytes()).hexdigest()
trace = [
    tool.tool_name
    for delegation, member in zip(response.tools, response.member_responses, strict=True)
    for tool in [delegation, *(member.tools or [])]
]
print(
    json.dumps(
        {"team": team.name, "tools": trace, **observed, "test_rows": len(rows), "test_mae": mae},
        indent=2,
    )
)
