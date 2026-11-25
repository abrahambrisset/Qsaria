import math

import pandas as pd
import pytest

from scripts.benchmark_qsaria_v1_moleculeace import read_official_data, score_cohort


def test_official_loader_keeps_split_assignments_and_cliff_flags(tmp_path):
    path = tmp_path / "endpoint.csv"
    path.write_text(
        "smiles,exp_mean [nM],y,cliff_mol,split\n"
        "CCO,100,-2,1,train\n"
        "CCO,10,-1,0,test\n"
        "CCN,1,0,1,test\n"
    )

    data = read_official_data(path)

    assert data.loc[data.split == "train", ["row_id", "Y", "cliff_mol"]].values.tolist() == [
        [0, -2, 1]
    ]
    assert data.loc[data.split == "test", ["row_id", "Y", "cliff_mol"]].values.tolist() == [
        [1, -1, 0],
        [2, 0, 1],
    ]


def test_official_loader_rejects_a_non_binary_flag(tmp_path):
    path = tmp_path / "endpoint.csv"
    path.write_text("smiles,y,cliff_mol,split\nCCO,-2,1,train\nCCN,-1,2,test\n")

    with pytest.raises(ValueError, match="cliff_mol"):
        read_official_data(path)


def test_rmse_cliff_uses_only_officially_flagged_rows():
    cohort = pd.DataFrame({"Y": [0.0, 0.0], "cliff_mol": [1, 0], "prediction": [2.0, 4.0]})

    report = score_cohort(cohort, {42: {"lightgbm": "prediction"}})

    assert report["n"] == 2
    assert report["n_cliff"] == 1
    assert report["by_seed"][42]["lightgbm"] == {"rmse": math.sqrt(10), "rmse_cliff": 2.0}
    assert (
        score_cohort(cohort.iloc[1:], {42: {"lightgbm": "prediction"}})["by_seed"][42]["lightgbm"][
            "rmse_cliff"
        ]
        is None
    )
