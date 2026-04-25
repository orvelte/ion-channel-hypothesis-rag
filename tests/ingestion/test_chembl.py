"""Unit tests for ChEMBL ingestion client."""

import pytest
from src.ingestion.chembl import ChEMBLCompound


def test_compound_model_valid():
    c = ChEMBLCompound(
        chembl_id="CHEMBL25",
        smiles="CC(=O)Nc1ccc(O)cc1",
        standard_value=10.5,
        standard_units="nM",
        standard_type="IC50",
        target_chembl_id="CHEMBL203",
        assay_chembl_id="CHEMBL123456",
        pchembl_value=8.0,
    )
    assert c.chembl_id == "CHEMBL25"


def test_compound_model_rejects_empty_smiles():
    with pytest.raises(Exception):
        ChEMBLCompound(
            chembl_id="CHEMBL25",
            smiles="   ",
            standard_value=10.5,
            standard_units="nM",
            standard_type="IC50",
            target_chembl_id="CHEMBL203",
            assay_chembl_id="CHEMBL123456",
        )


def test_fetch_not_implemented(tmp_path):
    from src.ingestion.chembl import fetch
    config = {
        "data": {"chembl_target_ids": ["CHEMBL203"]},
        "paths": {"rejected": str(tmp_path / "rejected")},
    }
    with pytest.raises(NotImplementedError):
        fetch(config, tmp_path)
