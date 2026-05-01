"""Invoice duplicate detection via LLM extraction of invoice number + amount."""

import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional

from sqlalchemy import select as sa_select, text

from app.models.duplicates import DuplicateInvoiceCache
from app.services.llm import LLMService

logger = logging.getLogger(__name__)


async def scan_invoices(
    paperless_client,
    session_factory,
    scan_state,
    llm_service: LLMService,
) -> List[Dict]:
    """Find duplicate invoices by extracting invoice number + amount via LLM.

    Updates scan_state.progress/total.
    """
    logger.info("Starting invoice duplicate scan")

    # Load all documents
    documents = await paperless_client.get_documents(page_size=100)

    # Load document types to resolve type names
    doc_types = await paperless_client.get_document_types()
    type_map: Dict[int, str] = {dt["id"]: dt.get("name", "") for dt in doc_types}

    # Filter for invoices
    invoice_docs = []
    for doc in documents:
        type_id = doc.get("document_type")
        type_name = type_map.get(type_id, "") if type_id else ""
        if type_name.lower() in ("rechnung", "invoice"):
            invoice_docs.append(doc)

    if not invoice_docs:
        logger.info("No invoice documents found, skipping invoice scan")
        return []

    scan_state.total = len(invoice_docs)
    logger.info(f"Found {len(invoice_docs)} invoice documents to analyze")

    # Build correspondent map
    corr_map = await _build_correspondent_map(paperless_client)

    # Load LLM config
    chat_provider, chat_model = await _get_chat_provider_and_model(session_factory)

    # Load cached extractions
    cached: Dict[int, Dict] = {}
    async with session_factory() as db:
        rows = (await db.execute(
            sa_select(DuplicateInvoiceCache)
        )).scalars().all()
        for row in rows:
            cached[row.document_id] = {
                "invoice_number": row.invoice_number,
                "amount": row.amount,
            }

    # Extract invoice data
    extractions: Dict[int, Dict] = {}  # doc_id -> {invoice_number, amount}

    for i, doc in enumerate(invoice_docs):
        if scan_state.is_cancelled():
            logger.info("Invoice scan cancelled by user")
            break

        scan_state.progress = i + 1
        doc_id = doc["id"]

        # Use cache if available
        if doc_id in cached:
            extractions[doc_id] = cached[doc_id]
            continue

        # Extract via LLM
        content = doc.get("content", "") or ""
        if not content.strip():
            continue

        # Truncate content to avoid huge prompts
        content_trimmed = content[:3000]

        extraction = await _extract_invoice_data(
            content_trimmed, chat_model, chat_provider, llm_service, session_factory
        )
        if extraction:
            extractions[doc_id] = extraction
            # Cache result
            await _cache_extraction(session_factory, doc_id, extraction)

    # Group by invoice_number
    number_groups: Dict[str, List[Dict]] = defaultdict(list)
    for doc in invoice_docs:
        doc_id = doc["id"]
        ext = extractions.get(doc_id)
        if not ext or not ext.get("invoice_number"):
            continue
        corr_id = doc.get("correspondent")
        number_groups[ext["invoice_number"]].append({
            "id": doc_id,
            "title": doc.get("title", ""),
            "created": doc.get("created", ""),
            "correspondent_name": corr_map.get(corr_id, "") if corr_id else "",
            "amount": ext.get("amount", ""),
        })

    # Filter: only groups where number AND amount match, and >1 doc
    results = []
    for inv_number, docs in number_groups.items():
        if len(docs) < 2:
            continue
        # Sub-group by amount
        amount_groups: Dict[str, List[Dict]] = defaultdict(list)
        for d in docs:
            amount_key = (d.get("amount") or "").strip().lower()
            amount_groups[amount_key].append(d)

        for amount_val, amount_docs in amount_groups.items():
            if len(amount_docs) > 1:
                results.append({
                    "invoice_number": inv_number,
                    "amount": amount_val,
                    "documents": amount_docs,
                })

    logger.info(f"Invoice scan found {len(results)} duplicate groups")
    return results


async def _build_correspondent_map(paperless_client) -> Dict[int, str]:
    """Build a {id: name} map of correspondents."""
    correspondents = await paperless_client.get_correspondents()
    return {c["id"]: c.get("name", "") for c in correspondents}


async def _get_chat_provider_and_model(session_factory) -> tuple[str, str]:
    """Get the configured chat provider and model from KV store."""
    from app.services.settings_service import get_setting
    async with session_factory() as db:
        provider = await get_setting("duplicate_chat_provider", db)
        model = await get_setting("duplicate_chat_model", db)
    if not provider or not model:
        raise ValueError("duplicate_chat_provider and duplicate_chat_model must be configured")
    return provider, model


async def _extract_invoice_data(
    content: str,
    model: str,
    provider: str,
    llm_service: LLMService,
    session_factory,
) -> Optional[Dict]:
    """Extract invoice number and amount from document content via LiteLLM."""
    async with session_factory() as db:
        from app.services.settings_service import get_prompt
        prompt = await get_prompt("duplicate_invoice_extraction", db, variables={"CONTENT": content})

    try:
        result = await llm_service.complete(
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=60.0,
        )
        return _parse_invoice_json(result.content or "")

    except Exception as e:
        logger.error(f"Invoice extraction failed: {e}")
        return None


def _parse_invoice_json(text: str) -> Optional[Dict]:
    """Try to extract a JSON object from LLM response text."""
    # Try direct parse
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return {
                "invoice_number": str(obj.get("invoice_number", "")).strip(),
                "amount": str(obj.get("amount", "")).strip(),
            }
    except json.JSONDecodeError:
        pass

    # Try to find JSON in text
    match = re.search(r'\{[^}]+\}', text)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return {
                    "invoice_number": str(obj.get("invoice_number", "")).strip(),
                    "amount": str(obj.get("amount", "")).strip(),
                }
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not parse invoice JSON from LLM response: {text[:200]}")
    return None


async def _cache_extraction(session_factory, doc_id: int, extraction: Dict):
    """Save extraction result to SQLite cache."""
    async with session_factory() as db:
        # Upsert: delete old, insert new
        await db.execute(
            text("DELETE FROM duplicate_invoice_cache WHERE document_id = :doc_id"),
            {"doc_id": doc_id},
        )
        cache_entry = DuplicateInvoiceCache(
            document_id=doc_id,
            invoice_number=extraction.get("invoice_number", ""),
            amount=extraction.get("amount", ""),
        )
        db.add(cache_entry)
        await db.commit()