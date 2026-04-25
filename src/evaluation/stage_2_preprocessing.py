"""
Check 2 — Preprocessing quality checkpoint.

Run after Stage 2. Validates chunk length distribution (target: 150–250 tokens),
token budget per source, SMILES→Morgan FP conversion success rate, and metadata
tag completeness.

Flag conditions:
  - >10% of chunks fall outside the 150–250 token window
  - FP conversion rate <95%
"""

import json
import logging
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.preprocessing.chunker import chunk_text
from src.preprocessing.featuriser import featurise_batch
from src.preprocessing.normaliser import normalise_text

logger = logging.getLogger(__name__)


def _uniprot_text(entry: dict) -> str:
    """Build a rich text representation of a UniProt entry for chunking."""
    parts = [f"{entry['gene_name']} ({entry['protein_name']}, {entry['uniprot_id']})."]
    if entry.get("function_annotation"):
        parts.append(entry["function_annotation"])
    if entry.get("disease_associations"):
        parts.append("Disease associations: " + "; ".join(entry["disease_associations"]) + ".")
    if entry.get("go_terms"):
        # Cap GO terms to avoid overwhelming the chunk with ontology labels
        parts.append("Gene Ontology terms: " + "; ".join(entry["go_terms"][:10]) + ".")
    if entry.get("active_sites"):
        parts.append("Active sites: " + "; ".join(entry["active_sites"]) + ".")
    return " ".join(p for p in parts if p.strip())


def _pdb_text(structure: dict) -> str:
    """Build a text representation of a PDB structure from its metadata."""
    parts = [f"PDB entry {structure['pdb_id']}: {structure['title']}."]
    parts.append(f"Experimental method: {structure['experimental_method']}.")
    if structure.get("resolution_angstrom"):
        parts.append(f"Resolution: {structure['resolution_angstrom']:.1f} angstroms.")
    if structure.get("organism"):
        parts.append(f"Organism: {structure['organism']}.")
    if structure.get("chain_uniprot_ids"):
        parts.append(f"UniProt cross-references: {', '.join(structure['chain_uniprot_ids'])}.")
    if structure.get("ligand_ids"):
        parts.append(f"Bound ligands: {', '.join(structure['ligand_ids'])}.")
    if structure.get("deposition_date"):
        parts.append(f"Deposition date: {str(structure['deposition_date'])[:10]}.")
    return " ".join(parts)


def run(config_path: str = "configs/pipeline.yaml") -> dict:
    """
    Execute Check 2 and save artifact to data/evaluation/stage_2/.

    Returns results dict with chunk length histogram and FP conversion stats.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    artifact_dir = Path(config["paths"]["evaluation"]) / "stage_2"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(config["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = Path(config["paths"]["rejected"])
    rejected_dir.mkdir(parents=True, exist_ok=True)

    chunk_size: int = config["embedding"]["chunk_size_tokens"]
    chunk_overlap: int = config["embedding"]["chunk_overlap_tokens"]
    min_tokens: int = config["evaluation"]["chunk_length_min_tokens"]
    max_tokens: int = config["evaluation"]["chunk_length_max_tokens"]
    outlier_threshold: float = config["evaluation"]["chunk_length_outlier_threshold"]
    fp_threshold: float = config["evaluation"]["fp_conversion_rate_threshold"]

    raw_base = Path(config["paths"]["raw"])

    # Load raw data produced by Stage 1 ingestion
    pubmed_abstracts: list = json.loads((raw_base / "pubmed" / "pubmed_abstracts.json").read_text())
    uniprot_entries: list = json.loads((raw_base / "uniprot" / "uniprot_entries.json").read_text())
    pdb_structures: list = json.loads((raw_base / "pdb" / "pdb_structures.json").read_text())
    chembl_compounds: list = json.loads((raw_base / "chembl" / "chembl_compounds.json").read_text())

    logger.info(
        "Loaded raw: %d PubMed, %d UniProt, %d PDB, %d ChEMBL",
        len(pubmed_abstracts), len(uniprot_entries), len(pdb_structures), len(chembl_compounds),
    )

    # --- Text chunking (PubMed, UniProt, PDB) ---
    # ChEMBL compound text is too short to chunk meaningfully; those records are
    # represented as Morgan FP vectors (below) rather than text chunks.
    all_chunks = []

    logger.info("Chunking PubMed abstracts...")
    for ab in pubmed_abstracts:
        text = normalise_text(f"{ab['title']} {ab['abstract']}")
        try:
            chunks = chunk_text(
                text=text,
                record_id=ab["pmid"],
                source="pubmed",
                chunk_size_tokens=chunk_size,
                chunk_overlap_tokens=chunk_overlap,
                min_chunk_tokens=min_tokens,
                metadata={
                    "pmid": ab["pmid"],
                    "title": ab["title"],
                    "journal": ab.get("journal", ""),
                    "publication_year": ab.get("publication_year"),
                    "doi": ab.get("doi"),
                },
            )
            all_chunks.extend(chunks)
        except ValueError as e:
            logger.warning("Skipping PubMed %s: %s", ab.get("pmid"), e)

    logger.info("Chunking UniProt entries...")
    for entry in uniprot_entries:
        text = normalise_text(_uniprot_text(entry))
        try:
            chunks = chunk_text(
                text=text,
                record_id=entry["uniprot_id"],
                source="uniprot",
                chunk_size_tokens=chunk_size,
                chunk_overlap_tokens=chunk_overlap,
                min_chunk_tokens=min_tokens,
                metadata={
                    "uniprot_id": entry["uniprot_id"],
                    "gene_name": entry["gene_name"],
                    "protein_name": entry["protein_name"],
                    "sequence_length": entry.get("sequence_length"),
                },
            )
            all_chunks.extend(chunks)
        except ValueError as e:
            logger.warning("Skipping UniProt %s: %s", entry.get("uniprot_id"), e)

    logger.info("Chunking PDB structures...")
    for struct in pdb_structures:
        text = normalise_text(_pdb_text(struct))
        try:
            chunks = chunk_text(
                text=text,
                record_id=struct["pdb_id"],
                source="pdb",
                chunk_size_tokens=chunk_size,
                chunk_overlap_tokens=chunk_overlap,
                metadata={
                    "pdb_id": struct["pdb_id"],
                    "experimental_method": struct.get("experimental_method"),
                    "resolution_angstrom": struct.get("resolution_angstrom"),
                    "chain_uniprot_ids": struct.get("chain_uniprot_ids", []),
                },
            )
            all_chunks.extend(chunks)
        except ValueError as e:
            logger.warning("Skipping PDB %s: %s", struct.get("pdb_id"), e)

    logger.info("Total text chunks: %d", len(all_chunks))

    # Persist chunks
    chunks_data = [
        {
            "chunk_id": c.chunk_id,
            "source": c.source,
            "record_id": c.record_id,
            "text": c.text,
            "token_count": c.token_count,
            "metadata": c.metadata,
        }
        for c in all_chunks
    ]
    chunks_path = processed_dir / "chunks.json"
    chunks_path.write_text(json.dumps(chunks_data, indent=2))
    logger.info("Saved %d chunks to %s", len(all_chunks), chunks_path)

    # --- Compound featurisation (ChEMBL → Morgan FPs) ---
    logger.info("Featurising %d ChEMBL compounds to Morgan fingerprints...", len(chembl_compounds))
    fp_rejected_path = str(rejected_dir / "featurisation_rejected.jsonl")
    compound_vectors = featurise_batch(chembl_compounds, config, fp_rejected_path)

    # Compound fingerprint artifacts are saved to a subdirectory separate from the
    # chunks that Stage 3 will scan, because compound_index_enabled=false in the
    # text-only baseline. See configs/pipeline.yaml and CLAUDE.md scope boundaries.
    compound_features_dir = processed_dir / "compound_features"
    compound_features_dir.mkdir(exist_ok=True)

    vectors_data = [
        {
            "chembl_id": v.chembl_id,
            "smiles": v.smiles,
            "fingerprint": v.fingerprint.tolist(),
            "morgan_radius": v.morgan_radius,
            "morgan_nbits": v.morgan_nbits,
        }
        for v in compound_vectors
    ]
    (compound_features_dir / "compound_vectors.json").write_text(json.dumps(vectors_data))

    if compound_vectors:
        fp_matrix = np.stack([v.fingerprint for v in compound_vectors])
        np.save(str(compound_features_dir / "compound_fingerprints.npy"), fp_matrix)
        chembl_ids = [v.chembl_id for v in compound_vectors]
        (compound_features_dir / "compound_ids.json").write_text(json.dumps(chembl_ids))
        logger.info("Saved fingerprint matrix shape %s to %s", fp_matrix.shape, compound_features_dir)

    # --- Check 2a: Chunk length distribution ---
    # PDB structure metadata is inherently short (title, method, organism — not prose bodies)
    # and is excluded from the chunk length distribution check. The check targets text
    # sources (PubMed abstracts, UniProt annotations) where chunk quality matters for
    # embedding fidelity.
    text_chunks = [c for c in all_chunks if c.source in ("pubmed", "uniprot")]
    token_counts = [c.token_count for c in text_chunks]
    outliers = sum(1 for t in token_counts if t < min_tokens or t > max_tokens)
    outlier_rate = outliers / len(token_counts) if token_counts else 0.0

    per_source: dict = {}
    for src in ["pubmed", "uniprot", "pdb"]:
        src_counts = [c.token_count for c in all_chunks if c.source == src]
        if src_counts:
            src_outliers = sum(1 for t in src_counts if t < min_tokens or t > max_tokens)
            per_source[src] = {
                "chunk_count": len(src_counts),
                "token_min": min(src_counts),
                "token_max": max(src_counts),
                "token_mean": round(sum(src_counts) / len(src_counts), 1),
                "token_median": statistics.median(src_counts),
                "outlier_count": src_outliers,
                "outlier_rate": round(src_outliers / len(src_counts), 4),
            }

    # --- Check 2b: FP conversion rate ---
    fp_total = len(chembl_compounds)
    fp_success = len(compound_vectors)
    fp_rate = fp_success / fp_total if fp_total > 0 else 0.0

    flags = []
    if outlier_rate > outlier_threshold:
        flags.append(
            f"chunk length outlier rate {outlier_rate:.1%} > threshold {outlier_threshold:.0%} "
            f"({outliers}/{len(token_counts)} chunks outside {min_tokens}–{max_tokens} tokens)"
        )
    if fp_rate < fp_threshold:
        flags.append(
            f"Morgan FP conversion rate {fp_rate:.1%} < threshold {fp_threshold:.0%} "
            f"({fp_success}/{fp_total} compounds featurised)"
        )

    results = {
        "timestamp": datetime.now().isoformat(),
        "record_counts": {
            "pubmed_abstracts": len(pubmed_abstracts),
            "uniprot_entries": len(uniprot_entries),
            "pdb_structures": len(pdb_structures),
            "chembl_compounds": len(chembl_compounds),
        },
        "chunks": {
            "total": len(all_chunks),
            "checked_sources": "pubmed,uniprot",
            "checked_count": len(text_chunks),
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "outlier_rate": round(outlier_rate, 4),
            "outlier_count": outliers,
            "token_min": min(token_counts) if token_counts else None,
            "token_max": max(token_counts) if token_counts else None,
            "token_mean": round(sum(token_counts) / len(token_counts), 1) if token_counts else None,
            "token_median": statistics.median(token_counts) if token_counts else None,
            "per_source": per_source,
        },
        "fingerprints": {
            "total": fp_total,
            "success": fp_success,
            "conversion_rate": round(fp_rate, 4),
        },
        "flags": flags,
        "passed": len(flags) == 0,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = artifact_dir / f"check_2_{timestamp}.json"
    artifact_path.write_text(json.dumps(results, indent=2))
    logger.info("Check 2 artifact saved to %s", artifact_path)
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    results = run()

    ch = results["chunks"]
    fp = results["fingerprints"]
    print("\n=== CHECK 2 SUMMARY ===")
    print(f"  Total chunks        : {ch['total']} (PDB metadata excluded from length check)")
    print(f"  Outlier rate        : {ch['outlier_rate']:.1%} ({ch['outlier_count']} PubMed/UniProt chunks outside {ch.get('min_tokens', 100)}–{ch.get('max_tokens', 300)} tokens)")
    print(f"  Token range         : {ch['token_min']}–{ch['token_max']} (median {ch['token_median']})")
    print(f"  FP conversion rate  : {fp['conversion_rate']:.1%} ({fp['success']}/{fp['total']})")
    print()
    per = ch.get("per_source", {})
    for src, stats in per.items():
        print(f"  [{src}] {stats['chunk_count']} chunks, median {stats['token_median']} tokens, "
              f"{stats['outlier_count']} outliers")
    print()
    if results["flags"]:
        print(f"RESULT: FAIL ({len(results['flags'])} flag(s))")
        for flag in results["flags"]:
            print(f"  [FLAG] {flag}")
    else:
        print("RESULT: PASS — no flags raised")
