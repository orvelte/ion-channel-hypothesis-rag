"""Unit tests for PDB ingestion client."""

import pytest
from src.ingestion.pdb import PDBStructure, fetch


def test_structure_model_valid():
    s = PDBStructure(
        pdb_id="6AGF",
        title="Structure of human Nav1.4",
        experimental_method="Cryo-EM",
        deposition_date="2018-08-01",
        resolution_angstrom=3.2,
    )
    assert s.pdb_id == "6AGF"


def test_fetch_not_implemented(tmp_path):
    config = {"data": {"pdb_max_structures": 5}}
    with pytest.raises(NotImplementedError):
        fetch(config, tmp_path)
