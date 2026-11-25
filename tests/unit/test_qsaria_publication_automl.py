import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def test_selection_runs_without_a_test_partition_and_evaluation_rejects_tampering(tmp_path):
    archive_path = tmp_path / "development.zip"
    train_rows = [f"drug_{index},{'C' * index},{index}\n" for index in range(1, 21)]
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "admet_group/example/train_val.csv", "Drug_ID,Drug,Y\n" + "".join(train_rows)
        )

    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    config = tmp_path / "config.toml"
    config.write_text(f"""archive_sha256 = "{archive_sha256}"
split_seed = 42
selection_seed = 42
test_seeds = [42]
radius = 2
bits = 32
trees = 4
workers = 1
bootstrap_repeats = 30
bootstrap_seed = 42

[datasets.example]
task = "regression"
metric = "mae"

[candidates.qsaria_lightgbm.gbdt]
learning_rate = 0.05
num_leaves = 7
min_child_samples = 2

[candidates.classical_automl.forest]
family = "random_forest"
max_features = "sqrt"
""")
    script = Path(__file__).resolve().parents[2] / "scripts/benchmark_qsaria_v1_automl.py"
    args = [sys.executable, str(script), "--config", str(config), "--archive", str(archive_path)]
    selections = [tmp_path / "select", tmp_path / "replay"]
    for output in selections:
        run = subprocess.run(
            [*args, "select", "--output", str(output)], capture_output=True, text=True, check=False
        )
        assert run.returncode == 0, run.stderr
    recipe = selections[0] / "selection.json"
    assert recipe.read_bytes() == (selections[1] / "selection.json").read_bytes()
    assert json.loads(recipe.read_bytes())["winners"]["example"] == {
        "qsaria_lightgbm": "gbdt",
        "classical_automl": "forest",
    }
    locked_sha256 = hashlib.sha256(recipe.read_bytes()).hexdigest()
    recipe.write_bytes(recipe.read_bytes() + b" ")

    evaluation_directory = tmp_path / "evaluate"
    run = subprocess.run(
        [
            *args,
            "evaluate",
            "--selection",
            str(recipe),
            "--selection-sha256",
            locked_sha256,
            "--output",
            str(evaluation_directory),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode != 0
    assert "selection checksum mismatch" in run.stderr.lower()
    assert not evaluation_directory.exists()
