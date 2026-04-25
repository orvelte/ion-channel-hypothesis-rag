"""
Text and chemical normalisation — Stage 2.

Normalisation is applied before chunking, not after, to prevent artefacts
(unicode noise, inconsistent gene-name capitalisation) from propagating into
embeddings. Gene name standardisation (SCN1A vs Nav1.1 vs NaV1.1) is important
here: the embedding model will not automatically align synonyms, so we collapse
them at the text level to improve known-pair retrieval in Check 3.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Synonym map: common aliases → canonical HGNC gene name
GENE_SYNONYMS: dict[str, str] = {
    "nav1.1": "SCN1A", "nav1.2": "SCN2A", "nav1.3": "SCN3A",
    "nav1.4": "SCN4A", "nav1.5": "SCN5A", "nav1.6": "SCN8A",
    "nav1.7": "SCN9A", "nav1.8": "SCN10A", "nav1.9": "SCN11A",
    "sodium channel alpha subunit 1": "SCN1A",
}

# Precompile patterns once at module load to avoid redundant re-compilation
_SYNONYM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(re.escape(alias), re.IGNORECASE), canonical)
    for alias, canonical in GENE_SYNONYMS.items()
]


def normalise_text(text: str) -> str:
    """
    Unicode-normalise, collapse whitespace, and standardise gene name synonyms.

    Does not lemmatise or stem — PubMedBERT is trained on raw biomedical text
    and handles morphological variation better than a stemmer would.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    for pattern, canonical in _SYNONYM_PATTERNS:
        text = pattern.sub(canonical, text)
    return text


def normalise_smiles(smiles: str) -> str:
    """
    Canonicalise SMILES via RDKit. Raises ValueError if RDKit cannot parse the input.
    RDKit canonical form is chosen over InChI for compatibility with Morgan FP generation.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True)
