import json

import pandas as pd
import pytest

from cs_copilot.tools.curation.dataset_curation_toolkit import DatasetCurationToolkit
from cs_copilot.tools.features.molecular_feature_toolkit import MolecularFeatureToolkit
from cs_copilot.tools.prediction.qsar_training_toolkit import QSARTrainingToolkit


@pytest.mark.parametrize(
    ("equivalent_structures", "expected_smiles"),
    [(["Oc1ccccn1", "O=c1cccc[nH]1"], "O=c1cccc[nH]1"), (["C[C@H](O)N", "C[C@@H](O)N"], "CC(N)O")],
)
def test_curation_merges_molecules_with_the_same_model_input(
    tmp_path, equivalent_structures, expected_smiles
):
    source = tmp_path / "source.csv"
    curated = tmp_path / "curated.csv"
    pd.DataFrame({"smiles": [*equivalent_structures, "CCO"], "y": [3.0, 3.4, 4.0]}).to_csv(
        source, index=False
    )

    result = DatasetCurationToolkit().curate_qsar_dataset(
        str(source), "regression", "smiles", ["y"], output_csv=str(curated)
    )

    rows = pd.read_csv(curated)
    assert result["rows_out"] == 2
    assert result["duplicate_groups_aggregated"] == 1
    assert rows["smiles"].tolist() == [expected_smiles, "CCO"]
    assert rows.iloc[0]["y"] == pytest.approx(3.2)
    assert rows["smiles"].equals(rows["curation_identity_key"])
    assert result["curation_policy"]["remove_stereochemistry"] is True
    manifest = json.loads(
        (tmp_path / "curated_curation_artifacts/curation_manifest.json").read_text()
    )
    assert manifest["curation_policy"]["remove_stereochemistry"] is True
    featurized = tmp_path / "features.csv"
    MolecularFeatureToolkit().smiles_to_morgan_fingerprints(
        str(curated), output_csv=str(featurized), n_bits=32, n_jobs=1
    )
    assert pd.read_csv(featurized)["smiles"].tolist() == rows["smiles"].tolist()


def test_training_refuses_an_old_curated_file_whose_distinct_tautomers_would_collapse(tmp_path):
    path = tmp_path / "old_curated.csv"
    old_identities = ["Oc1ccccn1", "O=c1cccc[nH]1", "CCO"]
    pd.DataFrame(
        {
            "smiles": old_identities,
            "qsar_identity_smiles": old_identities,
            "curation_identity_key": old_identities,
            "y": [2.0, 5.0, 7.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Recurate.*before splitting"):
        QSARTrainingToolkit()._prepare_tabular_training_dataset(
            train_csv=str(path),
            output_dir=str(tmp_path / "run"),
            smiles_column="smiles",
            target_columns=["y"],
            representation_name="morgan_only",
            feature_n_jobs=1,
        )
