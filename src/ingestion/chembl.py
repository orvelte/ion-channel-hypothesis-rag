"""
ChEMBL ingestion client — Stage 1.

Queries CNS compounds active on Na+/K+ channel assays identified by target
ChEMBL IDs (SCN1A–SCN9A). Direct HTTP calls to the ChEMBL REST API are used
instead of the chembl_webresource_client library, which contacts EBI at import
time and would crash before any error handling can run if EBI is unavailable.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel, field_validator
from rdkit import Chem

logger = logging.getLogger(__name__)

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 1000
STANDARD_TYPES = ["IC50", "Ki", "Kd", "EC50", "Inhibition", "Potency"]
REQUIRED_FIELDS = {
    "molecule_chembl_id", "canonical_smiles", "standard_value",
    "standard_units", "standard_type", "target_chembl_id", "assay_chembl_id",
}


class ChEMBLCompound(BaseModel):
    chembl_id: str
    smiles: str
    standard_value: float
    standard_units: str
    standard_type: str
    target_chembl_id: str
    assay_chembl_id: str
    pchembl_value: Optional[float] = None

    @field_validator("smiles")
    @classmethod
    def smiles_nonempty(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("SMILES must be non-empty")
        return v.strip()


def _append_rejected(rejected_path: Path, record: dict, reason: str) -> None:
    with open(rejected_path, "a") as f:
        f.write(json.dumps({"record": record, "reason": reason}) + "\n")


def _paginate_activities(target_id: str, session: requests.Session) -> list:
    """Fetch all activity records for a target via ChEMBL REST API pagination."""
    records = []
    params = {
        "target_chembl_id": target_id,
        "standard_type__in": ",".join(STANDARD_TYPES),
        "standard_relation": "=",
        "limit": PAGE_SIZE,
        "offset": 0,
        "format": "json",
    }
    while True:
        r = session.get(f"{CHEMBL_API}/activity.json", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        page = data.get("activities", [])
        records.extend(page)
        next_url = data.get("page_meta", {}).get("next")
        if not next_url or not page:
            break
        params["offset"] += PAGE_SIZE
        time.sleep(0.2)
    return records


# Morgan FPs are computed from the compounds returned here (in src/preprocessing/featuriser.py)
# but are NOT indexed in the Stage 3 vector store.
# Option 1 (text-only baseline) is active — see configs/pipeline.yaml retrieval.modality.
# If Check 4 reveals retrieval gaps on compound-specific queries,
# revisit Option 3 first: convert compound records to natural language
# chunks and embed as text. Do not build a second FAISS index until the
# text-only baseline is fully evaluated.
def fetch(config: dict, raw_dir: Path) -> list:
    """
    Fetch Na+/K+ channel compounds from ChEMBL and write raw JSON to raw_dir.

    Raises requests.HTTPError if the ChEMBL API is unreachable.
    Rejected records are appended to data/rejected/chembl_rejected.jsonl with reason.
    pChEMBL threshold and per-target cap are read from config to keep corpus ~500 compounds.
    """
    target_ids: list = config["data"]["chembl_target_ids"]
    pchembl_min: float = config["data"].get("chembl_pchembl_min", 5.0)
    max_per_target: int = config["data"].get("chembl_max_per_target", 150)
    rejected_path = Path(config["paths"]["rejected"]) / "chembl_rejected.jsonl"
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rejected_path.unlink(missing_ok=True)

    session = requests.Session()
    compounds: list = []
    seen: set = set()

    for target_id in target_ids:
        logger.info("Querying ChEMBL activities for %s (pChEMBL≥%.1f, cap=%d)",
                    target_id, pchembl_min, max_per_target)
        raw_records = _paginate_activities(target_id, session)
        logger.info("  %d raw records for %s", len(raw_records), target_id)

        target_compounds: list = []

        for record in raw_records:
            mol_id = record.get("molecule_chembl_id", "UNKNOWN")

            missing = {f for f in REQUIRED_FIELDS if not record.get(f)}
            if missing:
                _append_rejected(rejected_path, record, f"missing fields: {sorted(missing)}")
                continue

            # Require pChEMBL value — records without it lack reliable potency data
            try:
                pchembl = float(record["pchembl_value"])
            except (TypeError, ValueError):
                _append_rejected(rejected_path, record, "missing or non-numeric pchembl_value")
                continue
            if pchembl < pchembl_min:
                _append_rejected(rejected_path, record,
                                 f"pchembl_value {pchembl:.2f} < threshold {pchembl_min}")
                continue

            try:
                std_val = float(record["standard_value"])
            except (TypeError, ValueError):
                _append_rejected(rejected_path, record,
                                 f"non-numeric standard_value: {record['standard_value']!r}")
                continue

            if Chem.MolFromSmiles(record["canonical_smiles"]) is None:
                _append_rejected(rejected_path, record,
                                 f"RDKit unparseable SMILES: {record['canonical_smiles']!r}")
                continue

            dedup = (mol_id, record["target_chembl_id"], record["assay_chembl_id"], record["standard_type"])
            if dedup in seen:
                continue
            seen.add(dedup)

            target_compounds.append(ChEMBLCompound(
                chembl_id=mol_id,
                smiles=record["canonical_smiles"],
                standard_value=std_val,
                standard_units=record["standard_units"],
                standard_type=record["standard_type"],
                target_chembl_id=record["target_chembl_id"],
                assay_chembl_id=record["assay_chembl_id"],
                pchembl_value=pchembl,
            ))

        # Sort by potency (highest pChEMBL first), then cap
        target_compounds.sort(key=lambda c: -(c.pchembl_value or 0))
        accepted = target_compounds[:max_per_target]
        compounds.extend(accepted)
        logger.info("  %s: %d accepted (pChEMBL≥%.1f, cap=%d)",
                    target_id, len(accepted), pchembl_min, max_per_target)

    raw_file = raw_dir / "chembl_compounds.json"
    with open(raw_file, "w") as f:
        json.dump([c.model_dump() for c in compounds], f, indent=2)

    n_rejected = sum(1 for _ in open(rejected_path)) if rejected_path.exists() else 0
    logger.info("ChEMBL: %d compounds accepted, %d rejected", len(compounds), n_rejected)
    return compounds
