"""
PDB ingestion client — Stage 1.

Two-pass fetch strategy:
  1. UniProt cross-reference lookup — fetches PDB IDs directly linked to our human
     SCN UniProt accessions. This guarantees human Nav structures (cryo-EM / X-ray)
     are included even though they are fewer than bacterial homologue structures.
  2. Text search — captures prokaryotic structural templates (NavAb, NavMs) that
     dominate the text-search results. These are pharmacologically relevant: their
     high-resolution crystal structures define the gating mechanism vocabulary used
     in the primary literature, and they appear in abstracts we retrieve.

3D coordinate parsing is out of scope. Metadata only.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_ENTITY = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"

SOLVENT_IDS = {
    "HOH", "EDO", "GOL", "PEG", "SO4", "CL", "NA", "CA", "MG", "ZN",
    "K", "MN", "FE", "CU", "NI", "CO", "CD", "IOD", "BR", "F", "ACT",
}

TEXT_SEARCH_QUERY = {
    "query": {
        "type": "group",
        "logical_operator": "or",
        "nodes": [
            {"type": "terminal", "service": "full_text",
             "parameters": {"value": '"voltage-gated sodium channel"'}},
            {"type": "terminal", "service": "full_text",
             "parameters": {"value": '"Nav1" sodium channel'}},
        ],
    },
    "return_type": "entry",
    "request_options": {
        "paginate": {"start": 0, "rows": 60},
        "sort": [{"sort_by": "score", "direction": "desc"}],
    },
}


class PDBStructure(BaseModel):
    pdb_id: str
    title: str
    resolution_angstrom: Optional[float] = None
    experimental_method: str
    deposition_date: str
    organism: Optional[str] = None
    ligand_ids: list = []
    chain_uniprot_ids: list = []


def _fetch_human_pdb_ids(uniprot_ids: list, session: requests.Session) -> list:
    """
    Query UniProt cross-references to get PDB IDs for our human SCN proteins.

    Using UniProt as the lookup source (not RCSB) because UniProt maintains
    curated, reviewed cross-references, while RCSB's text-search returns
    prokaryotic structures preferentially due to higher coverage.
    """
    all_ids = []
    seen = set()
    for uid in uniprot_ids:
        try:
            r = session.get(
                f"{UNIPROT_BASE}/{uid}.json",
                params={"fields": "xref_pdb"},
                timeout=15,
            )
            if not r.ok:
                logger.warning("UniProt xref fetch failed for %s: HTTP %s", uid, r.status_code)
                continue
            pdb_ids = [
                ref["id"] for ref in r.json().get("uniProtKBCrossReferences", [])
                if ref.get("database") == "PDB"
            ]
            new = [p for p in pdb_ids if p not in seen]
            all_ids.extend(new)
            seen.update(new)
            logger.debug("UniProt %s → %d PDB IDs", uid, len(pdb_ids))
            time.sleep(0.2)
        except Exception as e:
            logger.warning("UniProt xref error for %s: %s", uid, e)
    return all_ids


def _text_search_pdb_ids(session: requests.Session) -> list:
    """Keyword search for voltage-gated sodium channel structures."""
    r = session.post(RCSB_SEARCH, json=TEXT_SEARCH_QUERY, timeout=30)
    r.raise_for_status()
    return [hit["identifier"] for hit in r.json().get("result_set", [])]


def _fetch_entry(pdb_id: str, session: requests.Session) -> Optional[PDBStructure]:
    r = session.get(RCSB_ENTRY.format(pdb_id=pdb_id), timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()

    title = data.get("struct", {}).get("title", "")
    if not title:
        return None

    resolution = None
    refine = data.get("refine", [])
    if refine:
        resolution = refine[0].get("ls_d_res_high")

    exptl = data.get("exptl", [{}])
    method = exptl[0].get("method", "UNKNOWN") if exptl else "UNKNOWN"
    dep_date = data.get("rcsb_accession_info", {}).get("deposit_date", "")

    ligands = [
        cc.get("id", "") for cc in data.get("chem_comp", [])
        if cc.get("id", "") and cc.get("id", "") not in SOLVENT_IDS
    ]

    entity_ids = data.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
    uniprot_ids = []
    organism = None
    for eid in entity_ids[:6]:
        try:
            er = session.get(RCSB_ENTITY.format(pdb_id=pdb_id, entity_id=eid), timeout=15)
            if er.ok:
                edata = er.json()
                refs = edata.get("rcsb_polymer_entity_container_identifiers", {})
                for uid in refs.get("uniprot_ids", []):
                    if uid not in uniprot_ids:
                        uniprot_ids.append(uid)
                if organism is None:
                    tax = edata.get("rcsb_entity_source_organism", [])
                    if tax:
                        organism = tax[0].get("scientific_name")
            time.sleep(0.1)
        except Exception as e:
            logger.debug("Entity fetch %s/%s: %s", pdb_id, eid, e)

    return PDBStructure(
        pdb_id=pdb_id,
        title=title,
        resolution_angstrom=float(resolution) if resolution is not None else None,
        experimental_method=method,
        deposition_date=str(dep_date),
        organism=organism,
        ligand_ids=ligands,
        chain_uniprot_ids=uniprot_ids,
    )


def fetch(config: dict, raw_dir: Path) -> list:
    """
    Fetch Na+/K+ channel structure metadata from RCSB PDB and write raw JSON to raw_dir.

    Human structures (UniProt xref) are fetched first and take priority in the cap.
    Bacterial structures from text search fill remaining slots. No 3D coordinate parsing.
    """
    max_structures: int = config["data"]["pdb_max_structures"]
    uniprot_ids: list = config["data"]["uniprot_ids"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    logger.info("Pass 1: fetching PDB IDs from UniProt cross-references (human SCN structures)")
    human_ids = _fetch_human_pdb_ids(uniprot_ids, session)
    logger.info("  %d human-linked PDB IDs found", len(human_ids))

    logger.info("Pass 2: text search for voltage-gated sodium channel structures (bacterial templates)")
    text_ids = _text_search_pdb_ids(session)
    logger.info("  %d text-search PDB IDs found", len(text_ids))

    # Human structures first, then bacterial; deduplicate
    seen = set()
    ordered_ids = []
    for pid in human_ids + text_ids:
        if pid not in seen:
            ordered_ids.append(pid)
            seen.add(pid)
    logger.info("  %d unique PDB IDs to fetch (capped at %d)", len(ordered_ids), max_structures)

    structures = []
    human_set = set(human_ids)
    for pdb_id in ordered_ids:
        if len(structures) >= max_structures:
            break
        try:
            s = _fetch_entry(pdb_id, session)
            if s:
                tag = "human" if pdb_id in human_set else "template"
                logger.info("PDB %s [%s]: %s (method=%s, res=%s Å)",
                            pdb_id, tag, s.title[:50], s.experimental_method,
                            f"{s.resolution_angstrom:.1f}" if s.resolution_angstrom else "N/A")
                structures.append(s)
        except Exception as e:
            logger.warning("Failed to fetch PDB %s: %s", pdb_id, e)
        time.sleep(0.2)

    (raw_dir / "pdb_structures.json").write_text(
        json.dumps([s.model_dump() for s in structures], indent=2)
    )

    human_count = sum(1 for s in structures if s.chain_uniprot_ids and
                      any(uid in set(uniprot_ids) for uid in s.chain_uniprot_ids))
    logger.info("PDB: %d structures total (%d with human UniProt xrefs)", len(structures), human_count)
    return structures
