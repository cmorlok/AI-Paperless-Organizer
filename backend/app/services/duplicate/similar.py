"""Similar document detection via ChromaDB embeddings (cosine similarity)."""

import asyncio
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")


async def scan_similar(
    paperless_client,
    session_factory,
    scan_state,
    similarity_threshold: float = 0.92,
) -> List[Dict]:
    """Find similar documents via ChromaDB embeddings (cosine similarity).

    Updates scan_state.progress/total.
    """
    logger.info(f"Starting similar document scan (threshold={similarity_threshold})")

    # Distance threshold: cosine distance = 1 - similarity
    distance_threshold = 1.0 - similarity_threshold

    persist_path = os.path.join(DATA_DIR, "chromadb")
    if not os.path.exists(persist_path):
        logger.warning("ChromaDB directory not found, skipping similar scan")
        return []

    import chromadb
    chroma_client = chromadb.PersistentClient(path=persist_path)
    try:
        collection = chroma_client.get_collection(name="paperless_documents")
    except Exception:
        logger.warning("ChromaDB collection 'paperless_documents' not found, skipping similar scan")
        return []

    # Get all embeddings for chunk_index=0 only
    all_data = collection.get(
        include=["embeddings", "metadatas"],
        where={"chunk_index": 0},
    )

    ids = all_data.get("ids", [])
    embeddings = all_data.get("embeddings", [])
    metadatas = all_data.get("metadatas", [])

    if len(ids) == 0 or embeddings is None or len(embeddings) == 0:
        logger.info("No embeddings found in ChromaDB for similar scan")
        return []

    scan_state.total = len(ids)
    logger.info(f"Querying {len(ids)} document embeddings for similarity")

    # Find similar pairs
    similar_pairs: List[Tuple[int, int, float]] = []  # (doc_id_a, doc_id_b, similarity)
    seen_pairs: Set[Tuple[int, int]] = set()

    # Build metadata lookup
    doc_meta: Dict[int, Dict] = {}
    if metadatas:
        for i, meta in enumerate(metadatas):
            doc_id = meta.get("document_id") if meta else None
            if doc_id is not None:
                doc_meta[int(str(doc_id))] = {
                    "id": int(str(doc_id)),
                    "title": str(meta.get("title", "")) if meta else "",
                    "chunk_id": ids[i],
                }

    # Batch query: process in chunks of 100 to avoid memory issues
    batch_size = 100
    total = len(embeddings)
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_embeddings = list(embeddings[batch_start:batch_end])
        batch_metadatas_slice = list(metadatas[batch_start:batch_end]) if metadatas else []

        scan_state.progress = batch_end

        results = collection.query(
            query_embeddings=list[Any](batch_embeddings),
            n_results=min(6, len(ids)),
        )

        if not results or not results.get("ids"):
            continue

        result_ids: list = results.get("ids") or []
        result_distances: list = results.get("distances") or []
        result_metas: list | None = results.get("metadatas")

        for i, (result_ids_row, distances_row) in enumerate(
            zip(result_ids, result_distances)
        ):
            meta_item = batch_metadatas_slice[i] if i < len(batch_metadatas_slice) else None
            doc_id_a = meta_item.get("document_id") if meta_item else None
            if doc_id_a is None:
                continue
            doc_id_a = int(str(doc_id_a))

            result_metas_row = result_metas[i] if result_metas and i < len(result_metas) else []

            for j, (rid, dist) in enumerate(zip(result_ids_row, distances_row)):
                doc_id_b = result_metas_row[j].get("document_id") if j < len(result_metas_row) and result_metas_row[j] else None
                if doc_id_b is None:
                    continue
                doc_id_b = int(str(doc_id_b))

                if doc_id_a == doc_id_b:
                    continue
                if dist > distance_threshold:
                    continue

                pair = (min(doc_id_a, doc_id_b), max(doc_id_a, doc_id_b))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                similarity = round(1.0 - dist, 4)
                similar_pairs.append((doc_id_a, doc_id_b, similarity))

        # Yield control + check cancellation between batches
        await asyncio.sleep(0)
        if scan_state.is_cancelled():
            logger.info("Similar scan cancelled by user")
            break

    # Group connected pairs into clusters
    clusters = _cluster_pairs(similar_pairs)

    # Build result groups with document info
    results = []
    for cluster_doc_ids, avg_similarity in clusters:
        docs = []
        for did in cluster_doc_ids:
            meta = doc_meta.get(did)
            if meta:
                docs.append({
                    "id": did,
                    "title": meta.get("title", ""),
                    "created": "",
                    "correspondent_name": "",
                })
            else:
                docs.append({
                    "id": did,
                    "title": f"Dokument #{did}",
                    "created": "",
                    "correspondent_name": "",
                })
        results.append({
            "similarity": avg_similarity,
            "documents": docs,
        })

    logger.info(f"Similar scan found {len(results)} groups")
    return results


def _cluster_pairs(pairs: List[Tuple[int, int, float]]) -> List[Tuple[List[int], float]]:
    """Group connected pairs into clusters using Union-Find.

    Returns list of (doc_id_list, average_similarity).
    """
    if not pairs:
        return []

    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Build union-find from pairs
    for doc_a, doc_b, _ in pairs:
        union(doc_a, doc_b)

    # Group by root
    cluster_members: Dict[int, List[int]] = defaultdict(list)
    cluster_sims: Dict[int, List[float]] = defaultdict(list)
    all_docs = set()
    for doc_a, doc_b, sim in pairs:
        all_docs.add(doc_a)
        all_docs.add(doc_b)

    for doc_id in all_docs:
        root = find(doc_id)
        if doc_id not in cluster_members[root]:
            cluster_members[root].append(doc_id)

    for doc_a, doc_b, sim in pairs:
        root = find(doc_a)
        cluster_sims[root].append(sim)

    results = []
    for root, members in cluster_members.items():
        if len(members) > 1:
            sims = cluster_sims.get(root, [0.0])
            avg_sim = round(sum(sims) / len(sims), 4) if sims else 0.0
            results.append((sorted(members), avg_sim))

    return results