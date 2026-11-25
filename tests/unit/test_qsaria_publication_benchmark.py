from zipfile import ZipFile

import numpy as np
import pandas as pd

from scripts.benchmark_qsaria_v1 import development_indices, paired_delta, read_partition


def test_development_archive_can_omit_the_official_test(tmp_path):
    archive_path = tmp_path / "development.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("admet_group/solubility_aqsoldb/train_val.csv", "Drug_ID,Drug,Y\na,CC,1\n")

    frame = read_partition(archive_path, "solubility_aqsoldb", "train_val")

    assert frame.loc[0, ["Drug_ID", "Drug", "Y"]].tolist() == ["a", "CC", 1]


def test_development_split_preserves_repeated_identity_groups():
    frame = pd.DataFrame({"identity": list("abcdef") * 2, "Y": range(12)})

    train, validation = development_indices(frame, "regression", 42)

    assert len(train) + len(validation) == len(frame)
    assert set(frame.iloc[train].identity).isdisjoint(frame.iloc[validation].identity)
    assert validation.tolist() == development_indices(frame, "regression", 42)[1].tolist()


def test_bootstrap_compares_both_models_on_the_same_identity_sample():
    truth = np.array([1.0, 3.0, 4.0, 8.0])
    prediction = np.array([1.0, 3.0, 4.0, 8.0])

    interval = paired_delta(truth, prediction, prediction, np.array(list("aabc")), "mae", 30, 42)

    assert interval == {"delta": 0.0, "lower_95": 0.0, "upper_95": 0.0, "valid": 30}
