"""
UniProt ingestion client — Stage 1.

Fetches canonical sequences and functional annotations for SCN1A–SCN9A via the
UniProt REST API. Canonical sequence only (no isoforms) keeps the corpus focused
and avoids embedding near-duplicate sequences that would inflate retrieval scores.
Functional annotations (active site, topology, disease variants) provide
structured context that text-only PubMed abstracts lack.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"


class UniProtEntry(BaseModel):
    uniprot_id: str
    gene_name: str
    protein_name: str
    organism: str
    sequence: str
    sequence_length: int
    function_annotation: Optional[str] = None
    active_sites: list = []
    disease_associations: list = []
    go_terms: list = []


def _get_with_retry(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            if attempt == retries - 1:
                raise
            logger.warning("HTTP error for %s (attempt %d), retrying", url, attempt + 1)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Unreachable: {url}")


def _parse_entry(data: dict) -> "UniProtEntry":
    uid = data.get("primaryAccession", "")
    if not uid:
        raise ValueError("Missing primaryAccession")

    gene_name = ""
    for gene in data.get("genes", []):
        gene_name = gene.get("geneName", {}).get("value", "")
        if gene_name:
            break
    if not gene_name:
        raise ValueError(f"Missing gene name for {uid}")

    protein_name = ""
    desc = data.get("proteinDescription", {})
    rec = desc.get("recommendedName", {})
    if rec:
        protein_name = rec.get("fullName", {}).get("value", "")
    if not protein_name:
        for sn in desc.get("submittedNames", []):
            protein_name = sn.get("fullName", {}).get("value", "")
            if protein_name:
                break
    if not protein_name:
        raise ValueError(f"Missing protein name for {uid}")

    organism = data.get("organism", {}).get("scientificName", "")
    if not organism:
        raise ValueError(f"Missing organism for {uid}")

    seq_data = data.get("sequence", {})
    sequence = seq_data.get("value", "")
    if not sequence:
        raise ValueError(f"Missing sequence for {uid}")
    sequence_length = seq_data.get("length", len(sequence))

    function_annotation = None
    disease_associations = []
    for comment in data.get("comments", []):
        ctype = comment.get("commentType", "")
        if ctype == "FUNCTION" and function_annotation is None:
            texts = comment.get("texts", [])
            if texts:
                function_annotation = texts[0].get("value", "")
        elif ctype == "DISEASE":
            disease_name = comment.get("disease", {}).get("diseaseId", "")
            if disease_name:
                disease_associations.append(disease_name)

    active_sites = []
    for feature in data.get("features", []):
        if feature.get("type") == "Active site":
            loc = feature.get("location", {})
            pos = loc.get("start", {}).get("value", "?")
            desc_text = feature.get("description", "")
            active_sites.append(f"pos={pos}" + (f" ({desc_text})" if desc_text else ""))

    go_terms = []
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref.get("database") == "GO":
            props = {p.get("key"): p.get("value") for p in ref.get("properties", [])}
            term = props.get("GoTerm", "")
            if term:
                go_terms.append(term)

    return UniProtEntry(
        uniprot_id=uid,
        gene_name=gene_name,
        protein_name=protein_name,
        organism=organism,
        sequence=sequence,
        sequence_length=sequence_length,
        function_annotation=function_annotation,
        active_sites=active_sites,
        disease_associations=disease_associations,
        go_terms=go_terms,
    )


def fetch(config: dict, raw_dir: Path) -> list:
    """
    Fetch SCN1A–SCN9A entries from UniProt REST API and write raw JSON to raw_dir.

    Raises ValueError if a UniProt ID returns no entry or sequence is missing.
    Raises RuntimeError on HTTP errors after 3 retries.
    """
    uniprot_ids: list = config["data"]["uniprot_ids"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    for uid in uniprot_ids:
        logger.info("Fetching UniProt entry %s", uid)
        data = _get_with_retry(f"{UNIPROT_BASE}/{uid}.json")
        entry = _parse_entry(data)
        entries.append(entry)
        (raw_dir / f"{uid}.json").write_text(json.dumps(data, indent=2))
        time.sleep(0.3)

    (raw_dir / "uniprot_entries.json").write_text(
        json.dumps([e.model_dump() for e in entries], indent=2)
    )
    logger.info("UniProt: fetched %d entries", len(entries))
    return entries
