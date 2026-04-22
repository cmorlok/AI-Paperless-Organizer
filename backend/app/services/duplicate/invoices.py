"""Invoice duplicate detection via LLM extraction of invoice number + amount."""

import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional

from sqlalchemy import select as sa_select, text

from app.models.duplicates import DuplicateInvoiceCache
from app.models.rag import RagConfig
from app.services.llm.lock import acquire as ollama_acquire, release as ollama_release
from app.services.llm.service import llm_completion

logger = logging.getLogger(__name__)


async def scan_invoices(
    paperless_client,
    session_factory,
    scan_state: Dict,
) -> List[Dict]:
    """Find duplicate invoices by extracting invoice number + amount via LLM.

    Updates scan_state["phase"], scan_state["progress"], scan_state["total"].
    """
    from app.services.duplicate.state import _is_cancelled

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

    scan_state["total"] = len(invoice_docs)
    logger.info(f"Found {len(invoice_docs)} invoice documents to analyze")

    # Build correspondent map
    corr_map = await _build_correspondent_map(paperless_client)

    # Load LLM config
    chat_model = await _get_chat_model(session_factory)

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
        if _is_cancelled():
            logger.info("Invoice scan cancelled by user")
            break

        scan_state["progress"] = i + 1
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

        extraction = await _extract_invoice_data(content_trimmed, chat_model)
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


async def _get_chat_model(session_factory) -> str:
    """Get the configured chat model from RagConfig."""
    async with session_factory() as db:
        result = await db.execute(
            sa_select(RagConfig).where(RagConfig.id == 1)
        )
        config = result.scalar_one_or_none()
    if config and config.chat_model:
        return config.chat_model
    return "qwen3.5:4b"


async def _extract_invoice_data(content: str, model: str) -> Optional[Dict]:
    """Extract invoice number and amount from document content via LiteLLM."""
    got = await ollama_acquire("duplicates", timeout=120)
    if not got:
        logger.warning("Could not acquire OllamaLock for invoice extraction")
        return None

    try:
        prompt = (
            "Extrahiere aus dem folgenden Dokumenttext die Rechnungsnummer und den Gesamtbetrag.\n"
            "Antworte NUR mit einem JSON-Objekt im Format:\n"
            '{"invoice_number": "...", "amount": "..."}\n'
            "Wenn du keine Rechnungsnummer findest, setze den Wert auf einen leeren String.\n"
            "Wenn du keinen Betrag findest, setze den Wert auf einen leeren String.\n\n"
            f"Dokumenttext:\n{content}"
        )

        response = await llm_completion(
            model=model,
            provider="ollama",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            num_ctx=4096,
            timeout=60.0,
        )
        reply = response.choices[0].message.content or ""
        return _parse_invoice_json(reply)

    except Exception as e:
        logger.error(f"Invoice extraction failed: {e}")
        return None
    finally:
        ollama_release("duplicates")


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