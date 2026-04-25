"""
Check 1 — Data integrity checkpoint.

Run after Stage 1 ingestion. Validates schema conformance across all four sources,
measures null/missing field rates, checks ChEMBL–UniProt–PDB cross-reference
coverage, and verifies SMILES parseability via RDKit.

Flag conditions:
  - >5% null rate on any required field
  - <80% cross-reference coverage between sources
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from typing import Optional

import requests
import yaml
from rdkit import Chem

logger = logging.getLogger(__name__)

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


def _null_rate(records: list, field: str) -> float:
    if not records:
        return 1.0
    return sum(1 for r in records if not r.get(field)) / len(records)


def _chembl_target_uniprot_xrefs(target_id: str, our_uniprot_ids: set) -> Optional[bool]:
    """Query ChEMBL target API for UniProt cross-references. Returns True/False/None."""
    try:
        r = requests.get(f"{CHEMBL_API}/target/{target_id}.json", timeout=15)
        r.raise_for_status()
        td = r.json()
        for comp in td.get("target_components", []):
            for xref in comp.get("target_component_xrefs", []):
                if xref.get("xref_src_db") == "UniProt" and xref.get("xref_id") in our_uniprot_ids:
                    return True
        return False
    except Exception as e:
        logger.warning("ChEMBL target lookup failed %s: %s", target_id, e)
        return None


def run(config_path: str = "configs/pipeline.yaml") -> dict:
    """
    Execute Check 1 and save artifact to data/evaluation/stage_1/.

    Calls fetch() on all four sources, then runs integrity checks.
    Returns a results dict with pass/fail status per check and summary stats.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    artifact_dir = Path(config["paths"]["evaluation"]) / "stage_1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_base = Path(config["paths"]["raw"])

    from src.ingestion import chembl, uniprot, pdb, pubmed

    logger.info("=== Stage 1: Ingestion ===")

    chembl_compounds = []
    chembl_source_error = None
    try:
        chembl_compounds = chembl.fetch(config, raw_base / "chembl")
    except Exception as e:
        chembl_source_error = str(e)
        logger.error("ChEMBL ingestion failed (API unavailable): %s", e)

    uniprot_entries = uniprot.fetch(config, raw_base / "uniprot")
    pdb_structures = pdb.fetch(config, raw_base / "pdb")
    pubmed_abstracts = pubmed.fetch(config, raw_base / "pubmed")

    logger.info("=== Check 1: Integrity ===")
    max_null_rate = config["evaluation"]["max_null_rate"]
    min_xref = config["evaluation"]["min_cross_reference_coverage"]

    results = {
        "timestamp": datetime.now().isoformat(),
        "record_counts": {
            "chembl": len(chembl_compounds),
            "uniprot": len(uniprot_entries),
            "pdb": len(pdb_structures),
            "pubmed": len(pubmed_abstracts),
        },
        "source_errors": {},
        "checks": {},
        "flags": [],
    }
    if chembl_source_error:
        results["source_errors"]["chembl"] = chembl_source_error
        results["flags"].append(f"ChEMBL API unavailable: {chembl_source_error[:120]}")

    # --- Check 1a: Null rates ---
    null_rates = {}
    chembl_dicts = [c.model_dump() for c in chembl_compounds]
    for field in ["chembl_id", "smiles", "standard_value", "standard_units",
                  "standard_type", "target_chembl_id", "assay_chembl_id"]:
        nr = _null_rate(chembl_dicts, field)
        null_rates[f"chembl.{field}"] = round(nr, 4)
        if nr > max_null_rate:
            results["flags"].append(
                f"null rate chembl.{field} = {nr:.1%} > threshold {max_null_rate:.0%}"
            )

    uniprot_dicts = [e.model_dump() for e in uniprot_entries]
    for field in ["uniprot_id", "gene_name", "sequence", "protein_name", "organism"]:
        nr = _null_rate(uniprot_dicts, field)
        null_rates[f"uniprot.{field}"] = round(nr, 4)
        if nr > max_null_rate:
            results["flags"].append(
                f"null rate uniprot.{field} = {nr:.1%} > threshold {max_null_rate:.0%}"
            )

    results["checks"]["null_rates"] = null_rates

    # --- Check 1b: SMILES parseability ---
    smiles_list = [c.smiles for c in chembl_compounds]
    parse_failures = [s for s in smiles_list if Chem.MolFromSmiles(s) is None]
    smiles_parse_rate = (1.0 - len(parse_failures) / len(smiles_list)) if smiles_list else None
    results["checks"]["smiles_parse_rate"] = round(smiles_parse_rate, 4) if smiles_parse_rate is not None else "n/a (no ChEMBL data)"
    results["checks"]["smiles_parse_failure_examples"] = parse_failures[:5]
    if parse_failures:
        results["flags"].append(
            f"{len(parse_failures)} SMILES failed RDKit parse post-ingestion filter (should be 0)"
        )

    # --- Check 1c: Cross-reference coverage ChEMBL → UniProt ---
    our_uniprot_ids = {e.uniprot_id for e in uniprot_entries}
    target_ids = config["data"]["chembl_target_ids"]
    xref_hits = 0
    xref_details = {}
    chembl_xref_available = not bool(chembl_source_error)

    if chembl_xref_available:
        for tid in target_ids:
            result_bool = _chembl_target_uniprot_xrefs(tid, our_uniprot_ids)
            xref_details[tid] = result_bool
            if result_bool:
                xref_hits += 1
        chembl_xref_coverage = xref_hits / len(target_ids) if target_ids else 1.0
        results["checks"]["chembl_to_uniprot_coverage"] = round(chembl_xref_coverage, 4)
        results["checks"]["chembl_target_xref_details"] = xref_details
        if chembl_xref_coverage < min_xref:
            results["flags"].append(
                f"ChEMBL→UniProt cross-ref coverage = {chembl_xref_coverage:.1%} < {min_xref:.0%}"
            )
    else:
        results["checks"]["chembl_to_uniprot_coverage"] = "skipped (ChEMBL API unavailable)"

    # --- Check 1d: Cross-reference coverage PDB → UniProt ---
    pdb_with_xref = sum(
        1 for s in pdb_structures
        if any(uid in our_uniprot_ids for uid in s.chain_uniprot_ids)
    )
    pdb_xref_coverage = pdb_with_xref / len(pdb_structures) if pdb_structures else 0.0
    results["checks"]["pdb_to_uniprot_coverage"] = round(pdb_xref_coverage, 4)
    if pdb_xref_coverage < min_xref:
        results["flags"].append(
            f"PDB→UniProt cross-ref coverage = {pdb_xref_coverage:.1%} < {min_xref:.0%}"
        )

    # --- Check 1e: Schema summary (informational) ---
    results["checks"]["schema_summary"] = {
        "chembl_unique_molecules": len({c.chembl_id for c in chembl_compounds}),
        "chembl_unique_targets": len({c.target_chembl_id for c in chembl_compounds}),
        "chembl_standard_types": sorted({c.standard_type for c in chembl_compounds}),
        "uniprot_genes": [e.gene_name for e in uniprot_entries],
        "uniprot_with_function_annotation": sum(1 for e in uniprot_entries if e.function_annotation),
        "pdb_experimental_methods": sorted({s.experimental_method for s in pdb_structures}),
        "pdb_with_ligands": sum(1 for s in pdb_structures if s.ligand_ids),
        "pubmed_year_range": [
            min(a.publication_year for a in pubmed_abstracts) if pubmed_abstracts else None,
            max(a.publication_year for a in pubmed_abstracts) if pubmed_abstracts else None,
        ],
        "pubmed_with_mesh": sum(1 for a in pubmed_abstracts if a.mesh_terms),
    }

    results["passed"] = len(results["flags"]) == 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = artifact_dir / f"check_1_{timestamp}.json"
    artifact_path.write_text(json.dumps(results, indent=2))
    logger.info("Check 1 artifact saved to %s", artifact_path)
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    results = run()

    print("\n=== CHECK 1 SUMMARY ===")
    rc = results["record_counts"]
    print(f"  ChEMBL compounds : {rc['chembl']}")
    print(f"  UniProt entries  : {rc['uniprot']}")
    print(f"  PDB structures   : {rc['pdb']}")
    print(f"  PubMed abstracts : {rc['pubmed']}")
    print()
    checks = results["checks"]
    smiles_rate = checks.get("smiles_parse_rate", "n/a")
    chembl_cov = checks.get("chembl_to_uniprot_coverage", "n/a")
    pdb_cov = checks.get("pdb_to_uniprot_coverage", "n/a")
    print(f"  SMILES parse rate          : {smiles_rate if isinstance(smiles_rate, str) else f'{smiles_rate:.1%}'}")
    print(f"  ChEMBL→UniProt coverage    : {chembl_cov if isinstance(chembl_cov, str) else f'{chembl_cov:.1%}'}")
    print(f"  PDB→UniProt coverage       : {pdb_cov if isinstance(pdb_cov, str) else f'{pdb_cov:.1%}'}")
    print()
    if results["flags"]:
        print(f"RESULT: FAIL ({len(results['flags'])} flag(s))")
        for flag in results["flags"]:
            print(f"  [FLAG] {flag}")
    else:
        print("RESULT: PASS — no flags raised")
