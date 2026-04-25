"""
PubMed ingestion client — Stage 1.

Fetches 100–200 abstracts on voltage-gated channel pharmacology and neuro drug
discovery via the Entrez E-utilities API. Abstracts are the text backbone of the
retrieval corpus — they provide mechanism-level language that ChEMBL assay records
and UniProt annotations don't contain. Keeping the corpus small and thematically
tight avoids topic drift in embedding space.
"""

import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
MAX_ABSTRACTS = 200
BATCH_SIZE = 50

SEARCH_QUERY = (
    "(voltage-gated sodium channel[MeSH Terms] OR SCN1A[tiab] OR SCN9A[tiab] "
    "OR Nav1[tiab] OR sodium channel[tiab]) "
    "AND (pharmacology[MeSH Subheading] OR drug discovery[tiab] "
    "OR epilepsy[MeSH Terms] OR neuropathic pain[MeSH Terms])"
)


class PubMedAbstract(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: list = []
    journal: str
    publication_year: int
    mesh_terms: list = []
    doi: Optional[str] = None


def _search_pmids(query: str, max_results: int) -> list:
    # sort=relevance gives a cross-section of the literature instead of the default
    # "most recent" ordering, which caused all 200 results to be from 2025-2026.
    r = requests.get(ESEARCH_URL, params={
        "db": "pubmed", "term": query, "retmax": max_results,
        "retmode": "json", "sort": "relevance",
    }, timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def _parse_article(article) -> Optional[dict]:
    pmid = article.findtext(".//PMID")
    if not pmid:
        return None

    medline = article.find(".//MedlineCitation")
    art = medline.find("Article") if medline is not None else None
    if art is None:
        return None

    # Use itertext() instead of .text to capture inline markup (e.g. <sup>+</sup> for ions,
    # <i> for gene names) — .text alone silently drops text inside child elements.
    title_el = art.find(".//ArticleTitle")
    title = "".join(title_el.itertext()).strip() if title_el is not None else ""

    abstract_el = art.find(".//Abstract")
    if abstract_el is None:
        return None
    parts = ["".join(t.itertext()) for t in abstract_el.findall(".//AbstractText")]
    abstract = " ".join(p.strip() for p in parts if p.strip())
    if not abstract:
        return None

    authors = []
    for author in art.findall(".//Author"):
        last = author.findtext("LastName", default="")
        first = author.findtext("ForeName", default="")
        if last:
            authors.append(f"{last} {first}".strip())

    journal = (art.findtext(".//Journal/Title", default="") or
               art.findtext(".//Journal/ISOAbbreviation", default=""))

    year_text = (art.findtext(".//Journal/JournalIssue/PubDate/Year") or
                 art.findtext(".//Journal/JournalIssue/PubDate/MedlineDate") or "")
    try:
        year = int(year_text[:4])
    except (ValueError, TypeError):
        return None

    mesh_terms = [
        mh.findtext("DescriptorName", default="")
        for mh in (medline.findall(".//MeshHeadingList/MeshHeading") if medline is not None else [])
        if mh.findtext("DescriptorName")
    ]

    doi = None
    for art_id in article.findall(".//ArticleIdList/ArticleId"):
        if art_id.get("IdType") == "doi":
            doi = art_id.text

    return {
        "pmid": pmid, "title": title, "abstract": abstract,
        "authors": authors, "journal": journal, "publication_year": year,
        "mesh_terms": mesh_terms, "doi": doi,
    }


def fetch(config: dict, raw_dir: Path) -> list:
    """
    Fetch abstracts from PubMed via Entrez E-utilities and write raw JSON to raw_dir.

    Raises ValueError if an abstract body is empty (useless for retrieval).
    Rejected records written to data/rejected/pubmed_rejected.jsonl.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    rejected_path = Path(config["paths"]["rejected"]) / "pubmed_rejected.jsonl"
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.unlink(missing_ok=True)

    logger.info("Searching PubMed: %s", SEARCH_QUERY[:80] + "...")
    pmids = _search_pmids(SEARCH_QUERY, MAX_ABSTRACTS)
    logger.info("Found %d PMIDs", len(pmids))

    raw_records = []
    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i:i + BATCH_SIZE]
        logger.info("Fetching PubMed batch %d/%d", i // BATCH_SIZE + 1,
                    (len(pmids) + BATCH_SIZE - 1) // BATCH_SIZE)
        r = requests.get(EFETCH_URL, params={
            "db": "pubmed", "id": ",".join(batch),
            "retmode": "xml", "rettype": "abstract",
        }, timeout=60)
        r.raise_for_status()

        root = ET.fromstring(r.text)
        for article in root.findall(".//PubmedArticle"):
            try:
                rec = _parse_article(article)
                if rec:
                    raw_records.append(rec)
            except Exception as e:
                pmid = article.findtext(".//PMID", default="?")
                logger.debug("Parse error PMID %s: %s", pmid, e)
        time.sleep(0.5)

    abstracts = []
    for rec in raw_records:
        try:
            abstracts.append(PubMedAbstract(**rec))
        except Exception as e:
            with open(rejected_path, "a") as f:
                f.write(json.dumps({"record": rec, "reason": str(e)}) + "\n")

    (raw_dir / "pubmed_abstracts.json").write_text(
        json.dumps([ab.model_dump() for ab in abstracts], indent=2)
    )
    n_rejected = sum(1 for _ in open(rejected_path)) if rejected_path.exists() else 0
    logger.info("PubMed: %d abstracts accepted, %d rejected", len(abstracts), n_rejected)
    return abstracts
