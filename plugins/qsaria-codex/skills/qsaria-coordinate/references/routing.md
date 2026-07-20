# Coordinator routing

## Request classes

| User intent | Route | Experiment behavior | Report |
| --- | --- | --- | --- |
| List, inspect, or recommend persisted models | Coordinator using registry read tools | No experiment | Skip |
| Read experiment state or artifacts | Coordinator lifecycle tools | Open only explicit id | Skip |
| Curate a dataset only | Curation | Create on first write | Once if a scientific explanation is requested |
| Train a model | Curation, Training, Registry only if persistence is missing | Create on first write | Once at the end |
| Predict with a persisted model | Inference | Create on first write; no old experiment needed | Once for substantive results |
| Evaluate a labelled dataset | Inference | Create on first write | Once, including terminal evaluation evidence |
| Inspect an ensemble | Coordinator using registry read tools | No experiment | Skip |
| Create or evaluate an ensemble | Registry | Create on first write | Once for substantive scientific results |
| Resume an experiment | Relevant roles from state | Open the supplied id only | According to the requested work |

## Routing rules

- Do not dispatch all agents by default. Use the smallest route that satisfies the request.
- A delegated agent must record a handoff, so delegate only after an experiment exists. Perform standalone read-only consultations directly instead of creating an otherwise empty experiment.
- Training may persist a directly publishable model itself. Otherwise, a returned persistence plan makes Registry mandatory before Report unless the experiment was explicitly created with `persistence_policy=session_only`. The durable default materializes reusable models under `data/model_assets/internal`; `.files` remains experiment evidence.
- Before delegating Inference, resolve one exact `model_id` from the user's request or the read-only catalog tools. Pass that id in the mission; Inference never performs catalog selection.
- Do not dispatch Report between roles. Intermediate status is carried by structured handoffs.
- Never dispatch a different scientific role until the preceding role's structured handoff is present and verified.
- A request that changes scientific meaning requires user input. A harmless presentation default may be chosen and disclosed.
- Do not treat a new Codex task as a continuation. Persisted models are global; experiment state is reopened only by identifier.
- When a project agent has a technical orchestration failure, the coordinator may perform one bounded manual recovery mission with that role's existing deterministic MCP allowlist. This is a fallback for the same route, not a new scientific route.
- A failed manual recovery ends execution mode. Offer a GitHub Issue draft; never patch the repository or publish the issue implicitly.

## Standard LightGBM mission

For an unqualified standard Qsaria LightGBM request, delegate this exact
contract to Training:

- backend `lightgbm`;
- representation `rdkit_all`;
- validation protocol `standard_qsar` with no custom validation strategy;
- toolkit-default hyperparameter tuning, currently 50 requested trials;
- toolkit-default outlier analysis when scientifically eligible.

Do not translate `simple`, `simple prompt`, or concise-output wording into a
scientific downgrade. Only an explicit user request may disable tuning or
outlier analysis, and the mission must then call the workflow `baseline` or
`simplified`, not `standard`.

## Agent mission packet

Every delegated mission should contain:

- `experiment_id` when the mission writes or resumes state;
- one role and one bounded objective;
- exact input artifact, file, and model identifiers;
- explicit scientific constraints and user prohibitions;
- an explicit named workflow and its expected effective settings when a standard contract applies;
- report language;
- expected handoff status and evidence fields.
