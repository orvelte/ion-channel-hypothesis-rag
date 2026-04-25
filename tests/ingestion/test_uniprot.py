"""Unit tests for UniProt ingestion client."""

import pytest
from src.ingestion.uniprot import UniProtEntry, fetch


def test_entry_model_valid():
    entry = UniProtEntry(
        uniprot_id="P35498",
        gene_name="SCN1A",
        protein_name="Sodium channel protein type 1 subunit alpha",
        organism="Homo sapiens",
        sequence="MAEQPALLKGLKRSSSQETEKAEADLKAELQNQAK",
        sequence_length=35,
    )
    assert entry.gene_name == "SCN1A"


def test_fetch_not_implemented(tmp_path):
    config = {"data": {"uniprot_ids": ["P35498"]}}
    with pytest.raises(NotImplementedError):
        fetch(config, tmp_path)
