"""Unit tests for PubMed ingestion client."""

import pytest
from src.ingestion.pubmed import PubMedAbstract, fetch


def test_abstract_model_valid():
    ab = PubMedAbstract(
        pmid="12345678",
        title="Lamotrigine block of voltage-gated sodium channels",
        abstract="Lamotrigine inhibits Nav1.1 with an IC50 of 50 µM...",
        authors=["Smith J", "Doe A"],
        journal="J Pharmacol",
        publication_year=2020,
    )
    assert ab.pmid == "12345678"


def test_fetch_not_implemented(tmp_path):
    config = {}
    with pytest.raises(NotImplementedError):
        fetch(config, tmp_path)
