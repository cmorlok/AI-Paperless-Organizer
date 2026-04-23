"""Exact duplicate detection via checksum matching."""

import logging
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)


async def scan_exact(
    paperless_client,
    session_factory,
    scan_state,
) -> List[Dict]:
    """Group documents by SHA256 checksum. Groups with >1 doc are exact duplicates.

    Updates scan_state.progress/total.
    """
    logger.info("Starting exact duplicate scan (checksum)")

    # Load documents with progress updates
    documents = []
    page = 1
    while True:
        result = await paperless_client._request("GET", "/documents/", params={"page_size": 100, "page": page})
        if not result:
            break
        if page == 1:
            total_count = result.get("count", 0)
            scan_state.total = total_count
            logger.info(f"Loading {total_count} documents for checksum scan...")
        docs = result.get("results", [])
        if not docs:
            break
        documents.extend(docs)
        scan_state.progress = len(documents)
        if not result.get("next"):
            break
        page += 1

    scan_state.total = len(documents)
    logger.info(f"Loaded {len(documents)} documents for checksum scan")

    # Build lookup maps for correspondent names
    corr_map = await _build_correspondent_map(paperless_client)

    # Group by checksum
    checksum_groups: Dict[str, List[Dict]] = defaultdict(list)
    for i, doc in enumerate(documents):
        scan_state.progress = i + 1
        checksum = doc.get("checksum") or doc.get("archive_checksum") or ""
        if not checksum:
            continue
        corr_id = doc.get("correspondent")
        checksum_groups[checksum].append({
            "id": doc["id"],
            "title": doc.get("title", ""),
            "created": doc.get("created", ""),
            "correspondent_name": corr_map.get(corr_id, "") if corr_id else "",
        })

    # Filter to groups with more than 1 document
    results = []
    for checksum, docs in checksum_groups.items():
        if len(docs) > 1:
            results.append({
                "checksum": checksum,
                "documents": docs,
            })

    logger.info(f"Exact scan found {len(results)} duplicate groups")
    return results


async def _build_correspondent_map(paperless_client) -> Dict[int, str]:
    """Build a {id: name} map of correspondents."""
    correspondents = await paperless_client.get_correspondents()
    return {c["id"]: c.get("name", "") for c in correspondents}