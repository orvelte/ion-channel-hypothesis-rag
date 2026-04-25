"""
Compound featurisation — Stage 2.

Morgan fingerprints (radius=2, 2048-bit) are used as dense vectors for
compounds rather than embedding SMILES strings directly. SMILES-to-embedding
requires a specialised chemistry LM (ChemBERTa etc.) and adds complexity
without proportional gain for a corpus of ~500 compounds at this scale.
Morgan FPs capture circular substructure and are standard in hit-list
expansion tasks, so the retrieval semantics are pharmacologically meaningful.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CompoundVector:
    chembl_id: str
    smiles: str
    fingerprint: np.ndarray    # shape: (morgan_nbits,), dtype float32
    morgan_radius: int
    morgan_nbits: int


def featurise_compound(
    smiles: str,
    chembl_id: str,
    morgan_radius: int,
    morgan_nbits: int,
) -> CompoundVector:
    """
    Generate a Morgan fingerprint for a single compound.

    Raises ValueError if RDKit cannot parse the SMILES string.
    Rejected compounds must be logged and written to data/rejected/; do not silently drop.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES for {chembl_id}: {smiles!r}")

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=morgan_radius, nBits=morgan_nbits)
    arr = np.zeros(morgan_nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return CompoundVector(
        chembl_id=chembl_id,
        smiles=smiles,
        fingerprint=arr,
        morgan_radius=morgan_radius,
        morgan_nbits=morgan_nbits,
    )


def featurise_batch(
    compounds: list[dict],
    config: dict,
    rejected_path: str,
) -> list[CompoundVector]:
    """
    Featurise a list of compound dicts. Writes failures to rejected_path.
    Returns only successfully featurised compounds; raises if success rate < threshold.
    """
    radius: int = config["compounds"]["morgan_radius"]
    nbits: int = config["compounds"]["morgan_nbits"]
    fp_threshold: float = config["evaluation"]["fp_conversion_rate_threshold"]

    rejected = Path(rejected_path)
    rejected.parent.mkdir(parents=True, exist_ok=True)

    vectors: list[CompoundVector] = []
    n_failed = 0

    for c in compounds:
        chembl_id = c.get("chembl_id", "UNKNOWN")
        smiles = c.get("smiles", "")
        try:
            vectors.append(featurise_compound(smiles, chembl_id, radius, nbits))
        except ValueError as e:
            n_failed += 1
            with open(rejected, "a") as f:
                f.write(json.dumps({"chembl_id": chembl_id, "smiles": smiles, "reason": str(e)}) + "\n")

    total = len(compounds)
    success_rate = (total - n_failed) / total if total > 0 else 0.0

    if success_rate < fp_threshold:
        raise RuntimeError(
            f"Morgan FP conversion rate {success_rate:.1%} below threshold {fp_threshold:.0%} "
            f"({len(vectors)}/{total} compounds featurised)"
        )

    logger.info(
        "Featurised %d/%d compounds (%.1f%% success, %d rejected)",
        len(vectors), total, success_rate * 100, n_failed,
    )
    return vectors
