from __future__ import annotations

import pandas as pd

from cs_copilot.tools.features.molecular_feature_toolkit import MolecularFeatureToolkit


def _sample_feature_input(tmp_path):
    input_csv = tmp_path / "molecules.csv"
    pd.DataFrame(
        {
            "SMILES": ["CCO", "CCN"],
            "Y": [1.0, 2.0],
        }
    ).to_csv(input_csv, index=False)
    return input_csv


def _sample_curated_feature_input(tmp_path):
    input_csv = tmp_path / "curated_molecules.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN"],
            "Y": [1.0, 2.0],
        }
    ).to_csv(input_csv, index=False)
    return input_csv


def _sample_dual_smiles_feature_input(tmp_path):
    input_csv = tmp_path / "dual_smiles_molecules.csv"
    pd.DataFrame(
        {
            "SMILES": ["CCO", "CCN"],
            "smiles": ["C(C)O", "C(C)N"],
            "Y": [1.0, 2.0],
        }
    ).to_csv(input_csv, index=False)
    return input_csv


def test_morgan_output_keeps_normalized_smiles_for_future_joins(tmp_path):
    toolkit = MolecularFeatureToolkit()
    input_csv = _sample_feature_input(tmp_path)
    output_csv = tmp_path / "morgan.csv"

    toolkit.smiles_to_morgan_fingerprints(
        input_csv=str(input_csv),
        smiles_column="SMILES",
        output_csv=str(output_csv),
        input_columns_to_keep=["Y"],
    )

    output_df = pd.read_csv(output_csv)
    assert "smiles" in output_df.columns
    assert "Y" in output_df.columns


def test_morgan_accepts_original_smiles_name_on_curated_lowercase_dataset(tmp_path):
    toolkit = MolecularFeatureToolkit()
    input_csv = _sample_curated_feature_input(tmp_path)
    output_csv = tmp_path / "morgan.csv"

    result = toolkit.smiles_to_morgan_fingerprints(
        input_csv=str(input_csv),
        smiles_column="SMILES",
        output_csv=str(output_csv),
        input_columns_to_keep=["SMILES", "Y"],
    )

    output_df = pd.read_csv(output_csv)
    assert result["smiles_column"] == "smiles"
    assert result["source_smiles_column"] == "smiles"
    assert "smiles" in output_df.columns
    assert "SMILES" not in output_df.columns
    assert "Y" in output_df.columns


def test_morgan_keeps_single_canonical_smiles_when_alias_columns_coexist(tmp_path):
    toolkit = MolecularFeatureToolkit()
    input_csv = _sample_dual_smiles_feature_input(tmp_path)
    output_csv = tmp_path / "morgan.csv"

    toolkit.smiles_to_morgan_fingerprints(
        input_csv=str(input_csv),
        smiles_column="SMILES",
        output_csv=str(output_csv),
        input_columns_to_keep=["SMILES", "smiles", "Y"],
    )

    output_df = pd.read_csv(output_csv)
    assert list(output_df.columns).count("smiles") == 1
    assert "SMILES" not in output_df.columns
    assert "Y" in output_df.columns


def test_morgan_count_fingerprints_are_tabular_counts(tmp_path):
    toolkit = MolecularFeatureToolkit()
    input_csv = _sample_feature_input(tmp_path)
    output_csv = tmp_path / "morgan_count.csv"

    result = toolkit.smiles_to_morgan_fingerprints(
        input_csv=str(input_csv),
        smiles_column="SMILES",
        output_csv=str(output_csv),
        input_columns_to_keep=["Y"],
        n_bits=64,
        feature_prefix="cfp_",
        fingerprint_kind="count",
    )

    output_df = pd.read_csv(output_csv)
    feature_columns = [column for column in output_df.columns if column.startswith("cfp_")]
    assert result["fingerprint_kind"] == "count"
    assert len(feature_columns) == 64
    assert output_df[feature_columns].to_numpy().max() >= 1
    assert output_df[feature_columns].dtypes.apply(lambda dtype: dtype.kind).isin(["i", "u"]).all()


def test_rdkit_output_keeps_normalized_smiles_for_future_joins(tmp_path):
    toolkit = MolecularFeatureToolkit()
    input_csv = _sample_feature_input(tmp_path)
    output_csv = tmp_path / "rdkit.csv"

    toolkit.smiles_to_rdkit_descriptors(
        input_csv=str(input_csv),
        smiles_column="SMILES",
        output_csv=str(output_csv),
        descriptor_set="basic",
        input_columns_to_keep=["Y"],
    )

    output_df = pd.read_csv(output_csv)
    assert "smiles" in output_df.columns
    assert "Y" in output_df.columns


def test_rdkit_accepts_original_smiles_name_on_curated_lowercase_dataset(tmp_path):
    toolkit = MolecularFeatureToolkit()
    input_csv = _sample_curated_feature_input(tmp_path)
    output_csv = tmp_path / "rdkit.csv"

    result = toolkit.smiles_to_rdkit_descriptors(
        input_csv=str(input_csv),
        smiles_column="SMILES",
        output_csv=str(output_csv),
        descriptor_set="basic",
        input_columns_to_keep=["SMILES", "Y"],
    )

    output_df = pd.read_csv(output_csv)
    assert result["smiles_column"] == "smiles"
    assert result["source_smiles_column"] == "smiles"
    assert "smiles" in output_df.columns
    assert "SMILES" not in output_df.columns
    assert "Y" in output_df.columns
