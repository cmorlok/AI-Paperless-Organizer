"""OCR service orchestrator."""

from __future__ import annotations

import base64
import asyncio
import json
import logging
import time
import io
from pathlib import Path
from typing import Optional, Dict, List, Any

from PIL import Image
from pdf2image import convert_from_bytes

from app.services.llm import LLMService

from .state import (
    DEFAULT_OCR_MODEL,
    OcrState,
    OcrDocumentProgress,
    OcrCompareSlot,
    OcrCompareState,
    PageProgress,
    TAG_RUN_OCR,
    TAG_OCR_FINISH,
    TAG_OCR_REVIEW,
    TAG_OCR_ERROR,
    QUALITY_THRESHOLD,
    MAX_ERROR_COUNT,
)

# Import file operations from local modules
from .review import load_review_queue, save_review_queue
from .error import (
    increment_ocr_error,
    reset_ocr_error,
    load_ocr_error_list,
    save_ocr_error_list,
    load_ocr_error_counts,
    save_ocr_error_counts,
)
from .ignore import load_ocr_ignore_list, save_ocr_ignore_list, get_ocr_ignored_ids

# Raise PIL pixel limit for large PDF pages rendered at high DPI
Image.MAX_IMAGE_PIXELS = 500_000_000  # 500 megapixels (default is ~178MP)

logger = logging.getLogger(__name__)


class OcrService:
    """Vision OCR service."""

    def __init__(
        self,
        state: OcrState | None = None,
        llm_service: LLMService | None = None,
        session_factory=None,
    ):
        self.state: OcrState = state or OcrState()
        self.llm_service = llm_service
        self.session_factory = session_factory
        self._prompts_registered = False

    async def _register_prompts(self, db: Any) -> None:
        """Register all OCR prompts with settings_service. Called once on first use."""
        from app.services.ocr.prompts import PROMPTS
        from app.services.settings_service import register_prompt
        for key, prompt_template in PROMPTS.items():
            await register_prompt(key, prompt_template, db)
        self._prompts_registered = True

    async def _get_prompt(self, key: str, variables: Optional[Dict[str, Any]] = None) -> str:
        """Get a prompt template by key, optionally rendered with variables."""
        from jinja2 import Template
        from app.services.ocr.prompts import PROMPTS
        from app.services.settings_service import get_prompt
        if self.session_factory is None:
            template_str = PROMPTS.get(key, "")
            if variables:
                return Template(template_str, autoescape=False).render(**variables)
            return template_str
        async with self.session_factory() as db:
            if not self._prompts_registered:
                await self._register_prompts(db)
            prompt_template = await get_prompt(key, db, variables)
            if prompt_template:
                return prompt_template
            template_str = PROMPTS.get(key, "")
            if variables:
                return Template(template_str, autoescape=False).render(**variables)
            return template_str

    async def _get_provider(self) -> str:
        assert self.llm_service is not None
        assert self.session_factory is not None
        """Read OCR provider from KV store."""
        from app.services.settings_service import get_setting
        async with self.session_factory() as db:
            provider = await get_setting("ocr_provider", db)
            if not provider:
                raise ValueError("ocr_provider is not configured")
            return provider

    async def _get_model(self) -> str:
        assert self.llm_service is not None
        assert self.session_factory is not None
        """Read OCR model from KV store."""
        from app.services.settings_service import get_setting
        async with self.session_factory() as db:
            return await get_setting("ocr_model", db) or DEFAULT_OCR_MODEL

    async def _get_max_image_size(self) -> int:
        assert self.llm_service is not None
        assert self.session_factory is not None
        """Lazy-load max image size from KV store."""
        from app.services.settings_service import get_setting
        async with self.session_factory() as db:
            val = await get_setting("max_image_size", db)
            return int(val) if val else 2048

    async def _get_smart_skip_enabled(self) -> bool:
        assert self.llm_service is not None
        assert self.session_factory is not None
        """Lazy-load smart skip setting from KV store."""
        from app.services.settings_service import get_setting
        async with self.session_factory() as db:
            val = await get_setting("smart_skip_enabled", db)
            return val != "false" if val else True

    async def _ensure_provider_ready(self, provider: str) -> None:
        assert self.llm_service is not None
        """Check that the OCR provider is accessible."""
        if self.llm_service is None:
            raise RuntimeError("OcrService.llm_service not injected - cannot perform OCR")
        ready = await self.llm_service.check_provider_health(provider)
        if not ready:
            raise ConnectionError(f"OCR provider '{provider}' ist nicht erreichbar")

    @staticmethod
    def get_model_params(model: str) -> Dict[str, Any]:
        """Return model-specific generation parameters for Ollama.

        Covers: qwen2.5vl, qwen2.5vl-7b-instruct-q3_K_S, qwen2.5vl-7b-instruct-q4_K_M,
                qwen2.5vl-7b-instruct-q5_K_M, qwen2.5vl-7b-instruct-fp16,
                qwen3-vl, qwen3-vl-7b, minicpm-v, minicpm-v-3b, minicpm-v-8b,
                deepseek-ocr, glm-ocr, gemma3.
        """
        name = (model or "").lower()

        if "deepseek-ocr" in name:
            return {
                "temperature": 0.2,
                "repeat_penalty": 1.1,
                "num_ctx": 4096,
                "num_predict": 8192,
                "max_image_size": 1280,
                "render_dpi": 200,
            }
        if "glm-ocr" in name or "glm_ocr" in name:
            return {
                "temperature": 0.1,
                "repeat_penalty": 1.05,
                "num_ctx": 4096,
                "num_predict": 8192,
                "max_image_size": 1536,
                "render_dpi": 200,
            }
        if "gemma3" in name or "gemma-3" in name:
            return {
                "temperature": 0.3,
                "repeat_penalty": 1.05,
                "num_ctx": 4096,
                "num_predict": 8192,
                "max_image_size": 896,
                "render_dpi": 200,
            }
        if "minicpm-v" in name:
            if "8b" in name:
                return {
                    "temperature": 0.3,
                    "repeat_penalty": 1.05,
                    "num_ctx": 4096,
                    "num_predict": 8192,
                    "max_image_size": 1280,
                    "render_dpi": 200,
                }
            return {
                "temperature": 0.3,
                "repeat_penalty": 1.05,
                "num_ctx": 4096,
                "num_predict": 4096,
                "max_image_size": 1024,
                "render_dpi": 200,
            }
        if "qwen3-vl" in name or "qwen3_vl" in name:
            return {
                "temperature": 0.2,
                "repeat_penalty": 1.05,
                "num_ctx": 4096,
                "num_predict": 8192,
                "max_image_size": 1280,
                "render_dpi": 200,
            }
        # Default: qwen2.5vl and similar 7b models
        return {
            "temperature": 0.2,
            "repeat_penalty": 1.05,
            "num_ctx": 4096,
            "num_predict": 8192,
            "max_image_size": 1280,
            "render_dpi": 200,
        }

    @staticmethod
    def _prepare_image(image: Image.Image, max_size: int = 1280) -> bytes:
        """Resize image to max dimension and convert to JPEG bytes for Ollama.

        Reduces image size to save tokens and speed up OCR without losing text legibility.
        """
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG", quality=85)
        return img_bytes.getvalue()

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """Strip <|begin_of_reasoning|>...<|end_of_reasoning|> blocks from text output.
        Inspired by paperless-gpt's stripReasoning approach."""
        import re
        cleaned = re.sub(r'<\|begin_of_reasoning\|>.*?<\|end_of_reasoning\|>', '', text, flags=re.DOTALL).strip()
        return cleaned if cleaned else text.strip()

    @staticmethod
    def _strip_thinking_text(text: str) -> str:
        """Strip residual thinking tags (e.g. <|thinking|>...<|/thinking|>) from text output."""
        import re
        cleaned = re.sub(r'<\|thinking\|>.*?<\|/thinking\|>', '', text, flags=re.DOTALL).strip()
        return cleaned if cleaned else text.strip()

    @staticmethod
    def _strip_ocr_commentary(text: str) -> str:
        """Remove trailing meta-commentary (e.g. 'Got it, let me transcribe...') from OCR output.
        Keeps only the actual transcribed document text."""
        if not text or len(text) < 100:
            return text
        lines = text.split("\n")
        # Patterns that start model commentary/description (English + German)
        commentary_starts = (
            "got it", "let me ", "let's ", "first,", "first i ", "i need to ", "starting from",
            "wait,", "let's check", "okay,", "so,", "now,", "i'll ", "i will ",
            "transcribe this", "go through each", "making sure not to miss",
            "die bild zeigt", "im bild steht", "zuerst muss ich"
        )
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            if not stripped:
                continue
            if any(stripped.startswith(p) or p in stripped[:50] for p in commentary_starts):
                # Keep everything before this line (the actual transcription)
                before = "\n".join(lines[:i]).strip()
                if len(before) > 80:
                    return before
        return text

    async def _build_ocr_prompt(self, model: str, page_num: int = 0, total_pages: int = 0) -> str:
        """Build OCR prompt using unified template with model-specific conditionals."""
        name = (model or "").lower()
        page_info = f" This is page {page_num} of {total_pages}." if page_num > 0 and total_pages > 0 else ""

        if "deepseek-ocr" in name:
            model_key = "deepseek"
        elif "glm-ocr" in name or "glm_ocr" in name:
            model_key = "glm"
        elif "gemma3" in name or "gemma-3" in name:
            model_key = "gemma3"
        else:
            model_key = "default"

        return await self._get_prompt("ocr_prompt", variables={"MODEL": model_key, "PAGE_INFO": page_info})

    @staticmethod
    def _clean_repetitions(text: str) -> str:
        """Detect and remove repetition loops from OCR output.

        Handles two cases:
        1. Instruction echo: model repeats the prompt/instruction 20+ times → truncate
        2. Content loops: same line appears many times → keep up to 6, skip the rest

        Conservative thresholds to avoid destroying real table data.
        """
        lines = text.split('\n')
        if len(lines) < 5:
            return text

        # Phase 1: Detect extreme instruction echo (20+ identical consecutive lines)
        run_start = 0
        last_stripped = ""
        run_count = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and stripped == last_stripped:
                run_count += 1
                if run_count >= 20:
                    lines = lines[:run_start]
                    logger.info(f"Instruction echo detected at line {run_start}, truncating")
                    break
            else:
                run_start = i
                run_count = 1 if stripped else 0
                last_stripped = stripped

        if len(lines) < 10:
            return "\n".join(lines).strip()

        # Phase 2: Collapse moderate repetitions (keep up to 6 identical consecutive lines)
        cleaned = []
        repeat_count = 0
        last_line = ""

        for line in lines:
            stripped = line.strip()
            if stripped == last_line and stripped:
                repeat_count += 1
                if repeat_count >= 6:
                    continue
            else:
                repeat_count = 0
            cleaned.append(line)
            last_line = stripped

        original_lines = len(lines)
        cleaned_lines = len(cleaned)
        if original_lines - cleaned_lines > 5:
            logger.info(f"Repetition cleanup: {original_lines} -> {cleaned_lines} lines (removed {original_lines - cleaned_lines} repeated lines)")

        return '\n'.join(cleaned)

    async def _ocr_single_image(
        self,
        image_bytes: bytes,
        model: str,
        provider: str,
        page_num: int = 0,
        total_pages: int = 0,
        timeout: float = 300.0,
    ) -> str | None:
        assert self.llm_service is not None
        """Run OCR on a single prepared image bytes block.

        Uses model-specific parameters from get_model_params().
        If a repetition loop is detected, retries with anti-loop parameters.
        """
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt_text = await self._build_ocr_prompt(model, page_num, total_pages)
        model_params = self.get_model_params(model)

        # First attempt with standard parameters
        text = await self._run_vision_ocr(
            image_b64, model, provider, prompt_text, model_params, timeout
        )

        if not text:
            return None

        # Detect repetition loop: if >60% of raw output was removed
        cleaned = text["_cleaned"] if isinstance(text, dict) else text
        loop_ratio = text.get("_loop_ratio", 0) if isinstance(text, dict) else 0

        if loop_ratio > 0.6:
            logger.info(f"Loop detected ({loop_ratio:.0%} wasted). Retrying with anti-table prompt...")
            retry_params = {**model_params}
            retry_params["num_predict"] = min(model_params["num_predict"], 4096)

            anti_table_prompt = await self._get_prompt("ocr_anti_table")

            retry_text = await self._run_vision_ocr(
                image_b64, model, provider, anti_table_prompt, retry_params, timeout
            )
            if retry_text:
                retry_cleaned = retry_text["_cleaned"] if isinstance(retry_text, dict) else retry_text
                retry_ratio = retry_text.get("_loop_ratio", 0) if isinstance(retry_text, dict) else 0

                if len(retry_cleaned) > len(cleaned):
                    logger.info(f"Anti-table retry improved: {len(cleaned)} -> {len(retry_cleaned)} chars (loop: {retry_ratio:.0%})")
                    return retry_cleaned
                else:
                    logger.info(f"Retry not better ({len(retry_cleaned)} vs {len(cleaned)} chars), keeping original")

        return cleaned

    async def _run_vision_ocr(
        self,
        image_b64: str,
        model: str,
        provider: str,
        prompt_text: str,
        model_params: dict,
        timeout: float,
    ) -> dict | str | None:
        assert self.llm_service is not None
        """Execute a single vision OCR request via LiteLLM."""
        if self.llm_service is None:
            raise RuntimeError("OcrService.llm_service not injected - cannot perform OCR")

        name_lower = (model or "").lower()
        use_think_param = "qwen3" in name_lower

        logger.debug(f"Model: {model}, repeat_pen={model_params['repeat_penalty']}, predict={model_params['num_predict']}")

        try:
            system_msg = await self._get_prompt("ocr_system_message")

            # LiteLLM multimodal format (OpenAI-compatible, works with Ollama vision)
            user_content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]

            result = await self.llm_service.complete(
                provider=provider,
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_content},
                ],
                temperature=model_params["temperature"],
                repeat_penalty=model_params["repeat_penalty"],
                num_ctx=model_params["num_ctx"],
                num_predict=model_params["num_predict"],
                keep_alive="30m",
                think=False if use_think_param else None,
                timeout=timeout,
            )

            text_content = (result.content or "").strip()
            text_content = self._strip_reasoning(text_content)
            text_content = self._strip_ocr_commentary(text_content)

            raw_len = len(text_content) if text_content else 0
            if text_content:
                text_content = self._clean_repetitions(text_content)
            cleaned_len = len(text_content) if text_content else 0
            loop_ratio = 1 - (cleaned_len / raw_len) if raw_len > 0 else 0

            if raw_len != cleaned_len:
                logger.info(f"Repetition cleanup: {raw_len} -> {cleaned_len} chars ({loop_ratio:.0%} removed)")

            return {
                "_cleaned": text_content,
                "_raw":     text_content,
                "_loop_ratio": loop_ratio,
            }

        except Exception as e:
            logger.error(f"Error during OCR request: {e}")
            return None

    def save_stats(self, doc_id: int, duration: float, pages: int, chars: int, success: bool = True, model: str = ""):
        """Save OCR statistics to JSON file."""
        from datetime import datetime

        stats_file = Path("/app/data/ocr_stats.json")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "duration": round(duration, 2),
            "pages": pages,
            "chars": chars,
            "model": model,
            "success": success
        }

        try:
            stats = []
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    try:
                        stats = json.load(f)
                    except Exception:
                        pass

            stats.append(entry)
            stats = stats[-1000:]  # Keep last 1000 entries

            with open(stats_file, "w") as f:
                json.dump(stats, f)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")

    def get_stats(self) -> List[Dict[str, Any]]:
        """Get OCR statistics."""
        stats_file = Path("/app/data/ocr_stats.json")
        if stats_file.exists():
            try:
                with open(stats_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _extract_text_from_pdf(self, file_bytes: bytes, smart_skip_enabled: bool = True) -> Optional[str]:
        """Extract text from PDF bytes using pypdf.

        Refined Logic (v1.1.7):
        - If text is found, check metadata.
        - If metadata indicates previous OCR (Abbyy, Tesseract, Paperless), return None (force new Vision OCR).
        - If metadata indicates 'Digital Born' (Word, LaTeX, Invoice Systems), return the text (Skip OCR).
        """
        if not smart_skip_enabled:
            logger.info("Smart-Skip disabled in settings. Forcing OCR.")
            return None

        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))

            meta = reader.metadata or {}
            creator = (meta.get("/Creator", "") or "").lower()
            producer = (meta.get("/Producer", "") or "").lower()

            ocr_keywords = ["abbyy", "finereader", "tesseract", "paperless", "ocr", "scan"]
            if any(k in creator for k in ocr_keywords) or any(k in producer for k in ocr_keywords):
                logger.info(f"Detected previous OCR tool in metadata ({creator} / {producer}). Forcing new OCR.")
                return None

            text = ""
            for i, page in enumerate(reader.pages):
                if i > 5:
                    break
                text += page.extract_text() + "\n\n"

            return text.strip()
        except Exception as e:
            logger.warning(f"Native PDF text extraction failed: {e}")
            return None

    async def ocr_document(self, paperless_client, document_id: int, force: bool = False, db_session=None) -> Dict[str, Any]:
        assert self.state is not None
        """OCR a document with page-level persistence. Supports resume after failures."""
        # Fetch provider config at runtime
        provider = await self._get_provider()
        model = await self._get_model()
        max_image_size = await self._get_max_image_size()
        smart_skip_enabled = await self._get_smart_skip_enabled()

        await self._ensure_provider_ready(provider)
        start_time = time.time()
        logger.info(f"Starting OCR for document {document_id}")

        self.state.page_progress[document_id] = OcrDocumentProgress(
            total_pages=0, done=0, errors=0,
            current_page=0, status="downloading",
            pages=[], started_at=time.time(),
        )

        doc = await paperless_client.get_document(document_id)
        if not doc:
            self.state.page_progress.pop(document_id, None)
            raise ValueError(f"Dokument {document_id} nicht gefunden")

        old_content = doc.get("content", "") or ""
        title = doc.get("title", f"Dokument {document_id}")

        MAX_FILE_SIZE_MB = 12
        try:
            file_bytes = await paperless_client.download_document_file(document_id)
            file_mb = len(file_bytes) / 1_048_576
            logger.info(f"Downloaded document {document_id}: {len(file_bytes)} bytes ({file_mb:.1f} MB)")

            if file_mb > MAX_FILE_SIZE_MB:
                self.state.page_progress.pop(document_id, None)
                error_msg = (
                    f"Dokument zu groß für OCR ({file_mb:.1f} MB > {MAX_FILE_SIZE_MB} MB). "
                    "Wird zur Ignore-Liste hinzugefügt."
                )
                ignore_list = load_ocr_ignore_list()
                if not any(entry["document_id"] == document_id for entry in ignore_list):
                    ignore_list.append({
                        "document_id": document_id, "title": title,
                        "reason": error_msg, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                    })
                    save_ocr_ignore_list(ignore_list)
                raise ValueError(error_msg)
        except ValueError:
            raise
        except Exception as e:
            self.state.page_progress.pop(document_id, None)
            if "404" in str(e):
                error_msg = "Originaldatei fehlt (404 Not Found)."
                ignore_list = load_ocr_ignore_list()
                if not any(entry["document_id"] == document_id for entry in ignore_list):
                    ignore_list.append({
                        "document_id": document_id, "title": title,
                        "reason": error_msg, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                    })
                    save_ocr_ignore_list(ignore_list)
                raise ValueError(f"{error_msg} Dokument wird künftig komplett ignoriert.")
            logger.error(f"Document download failed for {document_id}: {e}")
            raise ValueError("Download fehlgeschlagen")

        if not force:
            native_text = self._extract_text_from_pdf(file_bytes, smart_skip_enabled)
            if native_text and len(native_text) > 50:
                logger.info(f"Found native text in PDF ({len(native_text)} chars). Skipping OCR.")
                self.state.page_progress.pop(document_id, None)
                duration = time.time() - start_time
                return {
                    "document_id": document_id, "title": title,
                    "old_content": old_content, "new_content": native_text,
                    "old_length": len(old_content), "new_length": len(native_text),
                    "ocr_duration": duration, "ocr_pages": 0, "source": "native_pdf",
                }

        self.state.page_progress[document_id].status = "converting"
        images = await self._convert_to_images(file_bytes, doc, document_id, title, model)
        if not images:
            self.state.page_progress.pop(document_id, None)
            raise ValueError("Keine Seiten aus dem Dokument extrahiert")

        total_pages = len(images)
        self.state.page_progress[document_id].total_pages = total_pages
        self.state.page_progress[document_id].status = "processing"
        self.state.page_progress[document_id].pages = [PageProgress(page=i + 1) for i in range(total_pages)]

        completed_pages = {}
        if db_session:
            if force:
                await self._cleanup_page_results(db_session, document_id)
                logger.info(f"Force mode: cleared DB page cache for doc {document_id}, starting fresh")
            else:
                completed_pages = await self._load_completed_pages(db_session, document_id, total_pages)
                if completed_pages:
                    logger.info(f"Resume: found {len(completed_pages)} completed pages in DB")
                    for pg_num, pg_text in completed_pages.items():
                        idx = pg_num - 1
                        if idx < len(self.state.page_progress[document_id].pages):
                            self.state.page_progress[document_id].pages[idx] = PageProgress(
                                page=pg_num, status="done", chars=len(pg_text)
                            )
                    self.state.page_progress[document_id].done = len(completed_pages)

        MAX_PAGE_RETRIES = 3
        full_text_parts = {}
        failed_pages = []

        for i, img in enumerate(images):
            page_num = i + 1

            if page_num in completed_pages:
                full_text_parts[page_num] = completed_pages[page_num]
                logger.info(f"Page {page_num}/{total_pages}: resumed from DB ({len(completed_pages[page_num])} chars)")
                continue

            self.state.page_progress[document_id].current_page = page_num
            self.state.page_progress[document_id].pages[i].status = "processing"

            model_params = self.get_model_params(model)
            optimal_size = max(max_image_size, model_params["max_image_size"])
            prepared_bytes = self._prepare_image(img, max_size=optimal_size)

            page_text = None
            last_error = None

            for attempt in range(1, MAX_PAGE_RETRIES + 1):
                try:
                    page_start = time.time()
                    msg = f"Processing page {page_num}/{total_pages} (attempt {attempt})"
                    logger.info(msg)

                    page_text = await self._ocr_single_image(
                        prepared_bytes, model, provider, page_num=page_num, total_pages=total_pages
                    )

                    if not page_text or not page_text.strip():
                        raise ValueError("Empty result from OCR model")

                    page_duration = time.time() - page_start

                    if db_session:
                        await self._save_page_result(
                            db_session, document_id, page_num, total_pages,
                            page_text, "done", attempt, page_duration
                        )

                    full_text_parts[page_num] = page_text
                    self.state.page_progress[document_id].pages[i] = PageProgress(
                        page=page_num, status="done", chars=len(page_text)
                    )
                    self.state.page_progress[document_id].done += 1
                    logger.info(f"Page {page_num}: OK ({len(page_text)} chars, {page_duration:.1f}s)")
                    break

                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"OCR page {page_num} attempt {attempt} failed: {e}")

                    if attempt < MAX_PAGE_RETRIES:
                        await asyncio.sleep(2)

            if page_text is None:
                if db_session:
                    await self._save_page_result(
                        db_session, document_id, page_num, total_pages,
                        None, "error", MAX_PAGE_RETRIES, 0, last_error
                    )
                    self.state.page_progress[document_id].pages[i] = PageProgress(
                        page=page_num, status="error", chars=0, error=last_error
                    )
                    self.state.page_progress[document_id].errors += 1
                failed_pages.append(page_num)
                logger.warning(f"Page {page_num}: FAILED after {MAX_PAGE_RETRIES} attempts")

        if failed_pages:
            self.state.page_progress[document_id].status = "partial"
            raise ValueError(
                f"OCR fehlgeschlagen auf Seite(n) {failed_pages} von {total_pages}. "
                f"{len(full_text_parts)}/{total_pages} Seiten gespeichert -- "
                f"erneuter Versuch wird die fertigen Seiten wiederverwenden."
            )

        new_content = "\n\n".join(full_text_parts[p] for p in sorted(full_text_parts.keys()))
        duration = time.time() - start_time

        self.state.page_progress[document_id].status = "complete"

        if db_session:
            await self._cleanup_page_results(db_session, document_id)

        return {
            "document_id": document_id, "title": title,
            "old_content": old_content, "new_content": new_content,
            "old_length": len(old_content), "new_length": len(new_content),
            "ocr_duration": duration, "ocr_pages": total_pages,
        }

    async def _convert_to_images(self, file_bytes: bytes, doc: dict, document_id: int, title: str, model: str) -> list:
        """Convert file to list of PIL images."""
        model_params = self.get_model_params(model)
        render_dpi = model_params.get("render_dpi", 200)
        is_pdf = file_bytes[:4] == b'%PDF'
        mime_type = doc.get("mime_type", "unknown")

        if is_pdf:
            for attempt in range(2):
                try:
                    loop = asyncio.get_running_loop()
                    dpi = render_dpi if attempt == 0 else max(100, render_dpi - 50)
                    logger.warning(f"Converting PDF at {dpi} DPI (attempt {attempt+1})…")
                    images = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda d=dpi: convert_from_bytes(file_bytes, dpi=d)
                        ),
                        timeout=300
                    )
                    logger.info(f"Converted PDF to {len(images)} pages at {dpi} DPI")
                    return images
                except asyncio.TimeoutError:
                    logger.warning(f"PDF conversion timed out after 5 min (attempt {attempt+1})")
                    if attempt == 0:
                        logger.warning(f"PDF conversion timeout for doc {document_id}, retrying at lower DPI")
                        continue
                    raise ValueError("PDF-Konvertierung nach 10 Minuten abgebrochen – Dokument übersprungen.")
                except Exception as e:
                    error_str = str(e).lower()
                    if "password" in error_str or "encrypted" in error_str:
                        error_msg = "Passwortgeschützte PDF – kann ohne Passwort nicht verarbeitet werden."
                        ignore_list = load_ocr_ignore_list()
                        if not any(entry["document_id"] == document_id for entry in ignore_list):
                            ignore_list.append({
                                "document_id": document_id, "title": title,
                                "reason": error_msg, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                            })
                            save_ocr_ignore_list(ignore_list)
                        raise ValueError(f"{error_msg} Dokument wird künftig übersprungen.")
                    if attempt == 0:
                        logger.warning(f"PDF conversion failed (attempt 1), retrying: {e}")
                        await asyncio.sleep(2)
                    else:
                        logger.error(f"PDF conversion failed after 2 attempts ({mime_type}): {e}")
                        raise ValueError(f"PDF-Konvertierung fehlgeschlagen nach 2 Versuchen ({mime_type})")
        else:
            try:
                img = Image.open(io.BytesIO(file_bytes))
                img.load()
                logger.info(f"Loaded as single image ({img.format}, {img.size[0]}x{img.size[1]})")
                return [img]
            except Exception as e:
                logger.error(f"Image format not supported ({mime_type}, {len(file_bytes)} bytes): {e}")
                raise ValueError(f"Dateiformat nicht unterstützt ({mime_type})")
        return []

    async def _load_completed_pages(self, db_session, document_id: int, total_pages: int) -> Dict[int, str]:
        assert self.llm_service is not None
        """Load already-completed page results from the DB."""
        from sqlalchemy import select
        from app.models.ocr import OcrPageResult
        try:
            q = await db_session.execute(
                select(OcrPageResult).where(
                    OcrPageResult.document_id == document_id,
                    OcrPageResult.total_pages == total_pages,
                    OcrPageResult.status == "done",
                )
            )
            rows = q.scalars().all()
            return {r.page_number: r.page_text for r in rows if r.page_text}
        except Exception as e:
            logger.warning(f"Failed to load cached pages: {e}")
            return {}

    async def _save_page_result(
        self, db_session, document_id: int, page_number: int, total_pages: int,
        page_text: Optional[str], status: str, attempt_count: int,
        duration: float = 0, error_message: str | None = None,
    ):
        assert self.llm_service is not None
        """Upsert a page result into the DB."""
        from sqlalchemy import select
        from app.models.ocr import OcrPageResult
        try:
            q = await db_session.execute(
                select(OcrPageResult).where(
                    OcrPageResult.document_id == document_id,
                    OcrPageResult.page_number == page_number,
                )
            )
            row = q.scalars().first()
            if row:
                row.total_pages = total_pages
                row.page_text = page_text
                row.status = status
                row.attempt_count = attempt_count
                row.chars_extracted = len(page_text) if page_text else 0
                row.duration_seconds = duration
                row.error_message = error_message
            else:
                row = OcrPageResult(
                    document_id=document_id, page_number=page_number,
                    total_pages=total_pages, page_text=page_text,
                    status=status, attempt_count=attempt_count,
                    chars_extracted=len(page_text) if page_text else 0,
                    duration_seconds=duration, error_message=error_message,
                )
                db_session.add(row)
            await db_session.commit()
        except Exception as e:
            logger.error(f"Failed to save page result: {e}")
            try:
                await db_session.rollback()
            except Exception:
                pass

    async def _cleanup_page_results(self, db_session, document_id: int):
        assert self.llm_service is not None
        """Remove page results from DB after successful completion."""
        from sqlalchemy import delete
        from app.models.ocr import OcrPageResult
        try:
            await db_session.execute(
                delete(OcrPageResult).where(OcrPageResult.document_id == document_id)
            )
            await db_session.commit()
            logger.info(f"Cleaned up page cache for document {document_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup page results: {e}")

    async def ocr_image(self, image_bytes: bytes) -> str:
        assert self.llm_service is not None
        """Legacy method for backward compat or single image bytes."""
        try:
            provider = await self._get_provider()
            model = await self._get_model()
            max_size = await self._get_max_image_size()
            img = Image.open(io.BytesIO(image_bytes))
            prepared = self._prepare_image(img, max_size=max_size)
            result = await self._ocr_single_image(prepared, model, provider)
            return result or ""
        except Exception as e:
            logger.error(f"Legacy ocr_image failed: {e}")
            raise

    async def test_connection(self) -> Dict[str, Any]:
        assert self.llm_service is not None
        """Test OCR provider connection and return status."""
        try:
            provider = await self._get_provider()
            model = await self._get_model()
            connected = await self.llm_service.check_provider_health(provider)
            return {
                "connected": connected,
                "model": model,
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    async def get_ocr_status(self, paperless_client) -> Dict[str, Any]:
        """Get overall OCR status - total docs, finished docs, percentage."""
        ocrfinish_tag = await paperless_client.get_or_create_tag(TAG_OCR_FINISH)
        ocrfinish_id = ocrfinish_tag.get("id")

        total_count = await paperless_client.get_document_count()
        finished_count = await paperless_client.get_document_count(tag_id=ocrfinish_id) if ocrfinish_id else 0

        percentage = round((finished_count / total_count * 100), 1) if total_count > 0 else 0
        pending_count = total_count - finished_count

        return {
            "total_documents": total_count,
            "finished_documents": finished_count,
            "pending_documents": pending_count,
            "percentage": percentage,
            "ocrfinish_tag_id": ocrfinish_id,
        }

    async def apply_review_item(self, document_id: int, paperless_client) -> Dict[str, Any]:
        """Apply review queue item: accept the new OCR text for a document."""
        queue = load_review_queue()
        item = next((q for q in queue if q["document_id"] == document_id), None)
        if not item:
            raise ValueError("Dokument nicht in Review Queue")

        await self.apply_ocr_result(paperless_client, document_id, item["new_content"], True)

        queue = [q for q in queue if q["document_id"] != document_id]
        save_review_queue(queue)
        return {"applied": True, "document_id": document_id}

    async def reset_all_review_items(self, paperless_client) -> Dict[str, Any]:
        """Reset all review queue items: remove ocrpruefen tag so batch OCR re-processes them."""
        queue = load_review_queue()
        if not queue:
            return {"reset": 0, "errors": []}

        ocrpruefen_tag = await paperless_client.get_or_create_tag(TAG_OCR_REVIEW)
        ocrpruefen_id = ocrpruefen_tag.get("id")

        errors = []
        reset_count = 0
        for item in queue:
            doc_id = item["document_id"]
            try:
                if ocrpruefen_id:
                    await paperless_client.bulk_update_documents(
                        document_ids=[doc_id],
                        remove_tags=[ocrpruefen_id]
                    )
                reset_count += 1
            except Exception as e:
                errors.append(f"Dok {doc_id}: {e}")

        save_review_queue([])
        return {"reset": reset_count, "errors": errors}

    async def keep_all_originals(self, paperless_client) -> Dict[str, Any]:
        """Keep all original contents: set ocrfinish on all review items without changing content."""
        queue = load_review_queue()
        if not queue:
            return {"kept": 0, "errors": []}

        ocrfinish_tag = await paperless_client.get_or_create_tag(TAG_OCR_FINISH)
        ocrfinish_id = ocrfinish_tag.get("id")
        ocrpruefen_tag = await paperless_client.get_or_create_tag(TAG_OCR_REVIEW)
        ocrpruefen_id = ocrpruefen_tag.get("id")

        errors = []
        kept_count = 0
        doc_ids = [item["document_id"] for item in queue]

        for i in range(0, len(doc_ids), 25):
            batch = doc_ids[i:i+25]
            try:
                add_t = [ocrfinish_id] if ocrfinish_id else []
                rem_t = [ocrpruefen_id] if ocrpruefen_id else []
                if add_t or rem_t:
                    await paperless_client.bulk_update_documents(
                        document_ids=batch,
                        add_tags=add_t if add_t else None,
                        remove_tags=rem_t if rem_t else None
                    )
                kept_count += len(batch)
            except Exception as e:
                errors.append(f"Batch {i//25+1}: {e}")

        save_review_queue([])
        return {"kept": kept_count, "errors": errors}

    async def add_to_ignore_list(self, document_id: int, paperless_client) -> Dict[str, Any]:
        """Add a document to the OCR ignore list."""
        ignore_list = load_ocr_ignore_list()
        if any(entry["document_id"] == document_id for entry in ignore_list):
            return {"already_ignored": True, "document_id": document_id}

        title = f"Dokument {document_id}"
        try:
            doc = await paperless_client.get_document(document_id)
            if doc:
                title = doc.get("title", title)
        except Exception:
            pass

        ignore_list.append({
            "document_id": document_id,
            "title": title,
            "reason": "Original besser als OCR",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        })
        save_ocr_ignore_list(ignore_list)
        return {"added": True, "document_id": document_id, "title": title}

    async def remove_from_error_list(self, document_id: int, paperless_client) -> Dict[str, Any]:
        """Remove a document from the error list and remove its ocrfehler tag."""
        error_list = load_ocr_error_list()
        new_list = [entry for entry in error_list if entry["document_id"] != document_id]
        save_ocr_error_list(new_list)

        counts = load_ocr_error_counts()
        key = str(document_id)
        if key in counts:
            del counts[key]
            save_ocr_error_counts(counts)

        try:
            tag = await paperless_client.get_or_create_tag(TAG_OCR_ERROR)
            tag_id = tag.get("id")
            if tag_id:
                await paperless_client.bulk_update_documents(
                    document_ids=[document_id],
                    remove_tags=[tag_id]
                )
        except Exception as e:
            logger.warning(f"Could not remove ocrfehler tag from {document_id}: {e}")

        return {"removed": True, "document_id": document_id}

    async def evaluate_ocr_results(
        self,
        document_title: str,
        results: List[dict],
        eval_provider: str,
        eval_model: str | None = None,
    ) -> Dict[str, Any]:
        """Send OCR comparison results to an LLM for quality evaluation."""
        model_sections = []
        for i, r in enumerate(results):
            model_name = r.get("model", f"Modell {i+1}")
            text = r.get("text", "")
            chars = r.get("chars", len(text))
            duration = r.get("duration_seconds", 0)

            if len(text) > 6000:
                display_text = text[:4000] + "\n\n[... gekürzt ...]\n\n" + text[-1500:]
            else:
                display_text = text

            model_sections.append(
                f"=== VERSION {i+1}: {model_name} ===\n"
                f"Zeichen: {chars} | Dauer: {duration}s\n"
                f"--- TEXT START ---\n{display_text}\n--- TEXT END ---"
            )

        models_text = "\n\n".join(model_sections)

        prompt = await self._get_prompt(
            "ocr_evaluation",
            variables={
                "document_title": document_title,
                "version_count": len(results),
                "models_text": models_text,
            }
        )

        used_model = eval_model or "gpt-4o"
        logger.info("Evaluating OCR results", extra={"count": len(results), "provider": eval_provider, "model": used_model})

        assert self.llm_service is not None
        result = await self.llm_service.complete(
            provider=eval_provider,
            model=used_model,
            messages=[{"role": "user", "content": prompt}],
        )
        cleaned = (result.content or "").strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            evaluation = json.loads(cleaned)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match:
                evaluation = json.loads(json_match.group())
            else:
                logger.error(f"Could not parse LLM response as JSON: {cleaned[:500]}")
                return {
                    "success": True,
                    "raw_response": cleaned,
                    "evaluation": None,
                    "parse_error": "LLM-Antwort konnte nicht als JSON geparst werden",
                }

        logger.info("OCR evaluation complete", extra={"provider": eval_provider, "model": used_model})

        return {
            "success": True,
            "evaluation": evaluation,
            "provider": eval_provider,
            "model": used_model,
        }

    async def apply_ocr_result(
        self,
        paperless_client,
        document_id: int,
        new_content: str,
        set_finish_tag: bool = True
    ) -> Dict[str, Any]:
        assert self.state is not None
        """Apply OCR result to document and optionally set ocrfinish tag."""
        start_time = time.time()

        model = await self._get_model()

        await paperless_client.update_document(document_id, {"content": new_content})

        tag_success = True
        if set_finish_tag:
            tag = await paperless_client.get_or_create_tag(TAG_OCR_FINISH)
            tag_id = tag.get("id")
            if tag_id:
                for attempt in range(2):
                    try:
                        await paperless_client.bulk_update_documents(
                            document_ids=[document_id],
                            add_tags=[tag_id]
                        )
                        break
                    except Exception as e:
                        if attempt == 0:
                            logger.warning(f"Tag update for doc {document_id} failed, retrying: {e}")
                            await asyncio.sleep(2)
                        else:
                            tag_success = False
                            logger.error(f"Tag update for doc {document_id} failed after retry: {e}")

        duration = time.time() - start_time
        try:
            self.save_stats(document_id, duration, 0, len(new_content), success=tag_success, model=model)
        except Exception:
            pass

        return {"success": tag_success, "document_id": document_id}

    async def batch_ocr(
        self,
        paperless_client,
        mode: str = "all",
        document_ids: List[int] | None = None,
        set_finish_tag: bool = True,
        remove_runocr_tag: bool = True
    ) -> None:
        assert self.state is not None
        """Run batch OCR. Updates self.state.batch in-place for progress tracking."""
        self.state.batch.running = True
        self.state.batch.should_stop = False
        self.state.batch.total = 0
        self.state.batch.processed = 0
        self.state.batch.current_document = None
        self.state.batch.errors = []
        self.state.batch.log = []
        self.state.batch.mode = mode
        self.state.batch.paused = False

        model = await self._get_model()

        try:
            ocrfinish_tag = await paperless_client.get_or_create_tag(TAG_OCR_FINISH)
            ocrfinish_tag_id = ocrfinish_tag.get("id")

            ocrreview_tag = await paperless_client.get_or_create_tag(TAG_OCR_REVIEW)
            ocrreview_tag_id = ocrreview_tag.get("id")

            ocrerror_tag = await paperless_client.get_or_create_tag(TAG_OCR_ERROR)
            ocrerror_tag_id = ocrerror_tag.get("id")

            runocr_tag = None
            runocr_tag_id = None
            if mode == "tagged" or remove_runocr_tag:
                runocr_tag = await paperless_client.get_or_create_tag(TAG_RUN_OCR)
                runocr_tag_id = runocr_tag.get("id")

            documents = []

            # Provider config resolved at runtime - health check happens per-call in _ensure_provider_ready
            self.state.batch.log.append("🔎 Starte Batch-OCR (Provider-spezifische URL wird zur Laufzeit ermittelt)...")

            if mode == "all":
                self.state.batch.log.append("⏳ Lade Dokumentenliste von Paperless (das kann dauern)...")
                all_docs = None
                for _attempt in range(3):
                    try:
                        all_docs = await paperless_client.get_documents()
                        break
                    except Exception as fetch_err:
                        if _attempt < 2:
                            wait_sec = 30 * (_attempt + 1)
                            self.state.batch.log.append(
                                f"⚠️ Paperless nicht erreichbar (Versuch {_attempt + 1}/3): {fetch_err} – "
                                f"Retry in {wait_sec}s..."
                            )
                            logger.warning(f"batch_ocr: get_documents failed (attempt {_attempt+1}): {fetch_err}")
                            await asyncio.sleep(wait_sec)
                        else:
                            self.state.batch.log.append(
                                "❌ Paperless nach 3 Versuchen nicht erreichbar – Batch abgebrochen."
                            )
                            logger.error(f"batch_ocr: get_documents failed after 3 attempts: {fetch_err}")
                            return

                documents = [
                    d for d in (all_docs or [])
                    if ocrfinish_tag_id not in d.get("tags", [])
                    and ocrreview_tag_id not in d.get("tags", [])
                    and ocrerror_tag_id not in d.get("tags", [])
                ]
                self.state.batch.log.append(
                    f"📋 Modus: Alle Dokumente ({len(documents)} ohne ocrfinish/ocrpruefen/ocrfehler Tag)"
                )

            elif mode == "tagged":
                if runocr_tag_id:
                    documents = await paperless_client.get_documents(tag_id=runocr_tag_id)
                    documents = [
                        d for d in documents
                        if ocrfinish_tag_id not in d.get("tags", [])
                        and ocrreview_tag_id not in d.get("tags", [])
                        and ocrerror_tag_id not in d.get("tags", [])
                    ]
                self.state.batch.log.append(
                    f"🏷️ Modus: Nur mit Tag 'runocr' ({len(documents)} Dokumente)"
                )

            elif mode == "manual" and document_ids:
                for doc_id in document_ids:
                    try:
                        doc = await paperless_client.get_document(doc_id)
                        if doc and ocrfinish_tag_id not in doc.get("tags", []):
                            documents.append(doc)
                    except Exception:
                        self.state.batch.errors.append(f"Dokument {doc_id} nicht gefunden")
                self.state.batch.log.append(
                    f"✏️ Modus: Manuell ({len(documents)} Dokumente)"
                )

            ignored_ids = get_ocr_ignored_ids()
            if ignored_ids:
                before_count = len(documents)
                documents = [d for d in documents if d.get("id") not in ignored_ids]
                skipped = before_count - len(documents)
                if skipped > 0:
                    self.state.batch.log.append(f"🚫 {skipped} Dokument(e) übersprungen (OCR Ignore-Liste)")
                    logger.info(f"Skipped {skipped} ignored documents")

            self.state.batch.total = len(documents)

            if not documents:
                self.state.batch.log.append("⚠️ Keine Dokumente zum Verarbeiten gefunden.")
                return

            for i, doc in enumerate(documents):
                if self.state.batch.should_stop:
                    self.state.batch.log.append("🛑 Batch-OCR wurde gestoppt.")
                    break

                if self.state.batch.paused:
                    self.state.batch.log.append("⏸️ Batch-OCR pausiert...")
                    while self.state.batch.paused:
                        if self.state.batch.should_stop:
                            break
                        await asyncio.sleep(1)
                    if not self.state.batch.should_stop:
                        self.state.batch.log.append("▶️ Batch-OCR fortgesetzt.")

                if self.state.batch.should_stop:
                    break

                doc_id = doc.get("id")
                doc_title = doc.get("title", f"Dokument {doc_id}")
                self.state.batch.current_document = {"id": doc_id, "title": doc_title}
                self.state.batch.log.append(f"🔄 [{i+1}/{len(documents)}] Verarbeite: {doc_title} (ID: {doc_id})")

                try:
                    from app.database import async_session
                    async with async_session() as db_sess:
                        ocr_result = await self.ocr_document(paperless_client, doc_id, db_session=db_sess)
                    new_content = ocr_result.get("new_content")
                    old_content = ocr_result.get("old_content", "")
                    ocr_duration = ocr_result.get("ocr_duration", 0)
                    ocr_pages = ocr_result.get("ocr_pages", 1)
                    old_len = len(old_content) if old_content else 0
                    new_len = len(new_content) if new_content else 0

                    if new_content:
                        needs_review = False
                        if old_len > 100 and new_len < old_len * QUALITY_THRESHOLD:
                            ratio = round(new_len / old_len * 100) if old_len > 0 else 0
                            self.state.batch.log.append(
                                f"🔁 {doc_title}: Qualitätscheck fehlgeschlagen ({ratio}% des Originals) → Automatischer Retry..."
                            )
                            logger.warning(f"Quality check failed for {doc_id} ({ratio}%), retrying OCR...")

                            await asyncio.sleep(3)

                            try:
                                async with async_session() as db_sess2:
                                    retry_result = await self.ocr_document(paperless_client, doc_id, force=True, db_session=db_sess2)
                                retry_content = retry_result.get("new_content")
                                retry_len = len(retry_content) if retry_content else 0

                                if retry_content and retry_len > new_len:
                                    new_content = retry_content
                                    new_len = retry_len
                                    ocr_duration += retry_result.get("ocr_duration", 0)
                                    self.state.batch.log.append(
                                        f"🔁 {doc_title}: Retry lieferte besseres Ergebnis ({retry_len} vs {new_len - (retry_len - new_len)} Zeichen)"
                                    )
                                    logger.info(f"Retry improved: {retry_len} chars (was {new_len - (retry_len - new_len)})")
                                elif retry_content:
                                    if retry_len >= new_len:
                                        new_content = retry_content
                                        new_len = retry_len
                                    ocr_duration += retry_result.get("ocr_duration", 0)
                                    self.state.batch.log.append(
                                        f"🔁 {doc_title}: Retry ähnliches Ergebnis ({retry_len} Zeichen)"
                                    )
                                    logger.info(f"Retry similar: {retry_len} chars")
                            except Exception as retry_err:
                                self.state.batch.log.append(
                                    f"🔁 {doc_title}: Retry fehlgeschlagen - {str(retry_err)}"
                                )
                                logger.warning(f"Retry failed for {doc_id}: {retry_err}")

                            new_len = len(new_content) if new_content else 0
                            if old_len > 100 and new_len < old_len * QUALITY_THRESHOLD:
                                needs_review = True
                                ratio = round(new_len / old_len * 100) if old_len > 0 else 0
                                suggest_keep_original = old_len > 500 and ratio < 25
                                self.state.batch.log.append(
                                    f"⚠️ {doc_title}: Auch nach Retry nur {ratio}% des Originals "
                                    f"({new_len} vs {old_len} Zeichen) → In Prüfliste"
                                )
                                queue = load_review_queue()
                                queue.append({
                                    "document_id": doc_id,
                                    "title": doc_title,
                                    "old_content": old_content,
                                    "new_content": new_content,
                                    "old_length": old_len,
                                    "new_length": new_len,
                                    "ratio": ratio,
                                    "suggest_keep_original": suggest_keep_original,
                                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "retried": True
                                })
                                save_review_queue(queue)
                                try:
                                    add_t = [ocrreview_tag_id] if ocrreview_tag_id else []
                                    rem_t = [runocr_tag_id] if runocr_tag_id and remove_runocr_tag else []
                                    if add_t or rem_t:
                                        await paperless_client.bulk_update_documents(
                                            document_ids=[doc_id],
                                            add_tags=add_t if add_t else None,
                                            remove_tags=rem_t if rem_t else None
                                        )
                                        self.state.batch.log.append(f"🏷️ Tag 'ocrpruefen' an {doc_title} gehängt.")
                                except Exception as tag_err:
                                    logger.error(f"Failed to set ocrpruefen tag for {doc_id}: {tag_err}")
                            else:
                                self.state.batch.log.append(
                                    f"✅ {doc_title}: Retry erfolgreich! Qualität jetzt OK ({new_len} Zeichen)"
                                )
                                logger.info(f"Retry fixed quality for {doc_id}: {new_len} chars now passes threshold")

                        if not needs_review:
                            reset_ocr_error(doc_id)

                            await paperless_client.update_document(doc_id, {"content": new_content})

                            add_tags = []
                            remove_tags = []
                            tag_success = True

                            if set_finish_tag and ocrfinish_tag_id:
                                add_tags.append(ocrfinish_tag_id)
                            if remove_runocr_tag and runocr_tag_id:
                                remove_tags.append(runocr_tag_id)

                            if add_tags or remove_tags:
                                for attempt in range(2):
                                    try:
                                        await paperless_client.bulk_update_documents(
                                            document_ids=[doc_id],
                                            add_tags=add_tags if add_tags else None,
                                            remove_tags=remove_tags if remove_tags else None
                                        )
                                        tag_success = True
                                        break
                                    except Exception as e:
                                        tag_success = False
                                        if attempt == 0:
                                            logger.warning(f"Tag update for doc {doc_id} failed, retrying: {e}")
                                            await asyncio.sleep(2)
                                        else:
                                            logger.error(f"Tag update for doc {doc_id} failed after retry: {e}")
                                            self.state.batch.log.append(f"⚠️ {doc_title}: Tag-Update 2x fehlgeschlagen: {e}")

                            try:
                                self.save_stats(doc_id, ocr_duration, ocr_pages, new_len, success=tag_success, model=model)
                            except Exception:
                                pass

                            if not tag_success:
                                self.state.batch.log.append(f"⚠️ {doc_title}: Content OK, aber ocrfinish-Tag fehlt! ({new_len} Zeichen)")
                            else:
                                self.state.batch.log.append(f"✅ {doc_title}: OCR erfolgreich ({new_len} Zeichen)")
                    else:
                        self.state.batch.log.append(f"⚠️ {doc_title}: Kein Text erkannt")
                        self.state.batch.errors.append(f"{doc_title}: Kein Text erkannt")

                except Exception as e:
                    try:
                        self.save_stats(doc_id, 0, 0, 0, success=False, model=model)
                    except Exception:
                        pass
                    error_msg = f"❌ {doc_title}: Fehler - {str(e)}"
                    self.state.batch.log.append(error_msg)
                    self.state.batch.errors.append(error_msg)
                    logger.error(f"OCR error for document {doc_id}: {e}")

                    err_count = increment_ocr_error(doc_id, doc_title, str(e))
                    if err_count >= MAX_ERROR_COUNT:
                        try:
                            add_t = [ocrerror_tag_id] if ocrerror_tag_id else []
                            rem_t = [runocr_tag_id] if runocr_tag_id and remove_runocr_tag else []
                            if add_t or rem_t:
                                await paperless_client.bulk_update_documents(
                                    document_ids=[doc_id],
                                    add_tags=add_t if add_t else None,
                                    remove_tags=rem_t if rem_t else None
                                )
                            error_list = load_ocr_error_list()
                            if not any(entry["document_id"] == doc_id for entry in error_list):
                                error_list.append({
                                    "document_id": doc_id,
                                    "title": doc_title,
                                    "error": str(e)[:200],
                                    "fail_count": err_count,
                                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                                })
                                save_ocr_error_list(error_list)
                            self.state.batch.log.append(
                                f"🚫 {doc_title}: {err_count}x fehlgeschlagen → Tag 'ocrfehler' gesetzt, wird nicht mehr verarbeitet"
                            )
                            logger.error(f"Doc {doc_id} permanently marked as failed ({err_count} failures)")
                        except Exception as tag_err:
                            logger.error(f"Failed to set ocrfehler tag for {doc_id}: {tag_err}")
                    else:
                        self.state.batch.log.append(
                            f"⚠️ {doc_title}: Fehler {err_count}/{MAX_ERROR_COUNT} - wird beim nächsten Lauf erneut versucht"
                        )

                self.state.batch.processed = i + 1

                await asyncio.sleep(5)

            self.state.batch.log.append(
                f"🏁 Fertig! {self.state.batch.processed}/{self.state.batch.total} Dokumente verarbeitet, "
                f"{len(self.state.batch.errors)} Fehler."
            )

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.state.batch.log.append(f"💥 Kritischer Fehler: {str(e)}")
            self.state.batch.log.append(f"Traceback: {error_details}")
            logger.error(f"Batch OCR critical error: {e}\n{error_details}")
        finally:
            self.state.batch.running = False
            self.state.batch.current_document = None

    async def _unload_model_from_vram(self, model: str) -> None:
        assert self.llm_service is not None
        try:
            await self.llm_service.unload_local_model("ollama", model)
            logger.info(f"Unloaded {model} from VRAM")
        except Exception:
            pass

    async def _wait_for_provider_ready(self, provider: str, max_wait: int = 60) -> bool:
        assert self.llm_service is not None
        waited = 0
        interval = 3
        while waited < max_wait:
            try:
                if await self.llm_service.check_provider_health(provider):
                    if waited > 0:
                        logger.info(f"{provider} wieder erreichbar nach {waited}s Wartezeit")
                    return True
            except Exception:
                pass
            logger.warning(f"{provider} nicht erreichbar, warte {interval}s... ({waited}/{max_wait}s)")
            await asyncio.sleep(interval)
            waited += interval
        logger.warning(f"{provider} nach {max_wait}s immer noch nicht erreichbar!")
        return False

    async def run_compare_job(
        self,
        paperless_client,
        document_id: int,
        slots: list[OcrCompareSlot],
        target_page: int,
        compare_state: OcrCompareState,
    ) -> None:
        """Run an OCR model comparison job. Designed to be launched as a background task."""
        job_start = time.time()
        compare_state.job_start = job_start

        try:
            compare_state.phase = "download"
            doc = await paperless_client.get_document(document_id)
            if not doc:
                raise ValueError(f"Dokument {document_id} nicht gefunden")

            compare_state.title = doc.get("title", f"Dokument {document_id}")
            compare_state.old_content = doc.get("content", "") or ""

            file_bytes = await paperless_client.download_document_file(document_id)
            logger.info(f"Downloaded doc {document_id}: {len(file_bytes)} bytes")

            compare_state.phase = "convert"
            is_pdf = True
            try:
                loop = asyncio.get_running_loop()
                preview_images = await loop.run_in_executor(
                    None, lambda: convert_from_bytes(file_bytes, dpi=150)
                )
                total_pages = len(preview_images)
                logger.info(f"Document has {total_pages} pages")
            except Exception:
                is_pdf = False
                preview_images = [Image.open(io.BytesIO(file_bytes))]
                total_pages = 1

            if total_pages == 0:
                raise ValueError("Keine Seiten extrahiert")

            compare_state.total_pages = total_pages

            if target_page > 0 and target_page <= total_pages:
                page_indices = [target_page - 1]
                compare_state.compared_page = target_page
            else:
                page_indices = list(range(total_pages))
                compare_state.compared_page = 0

            dpi_image_cache: dict = {}

            for model_idx, slot in enumerate(slots):
                model_name = slot.model
                provider = slot.provider
                compare_state.current_model = model_name
                compare_state.current_model_index = model_idx
                compare_state.current_page = 0
                compare_state.elapsed_seconds = round(time.time() - job_start, 1)

                compare_state.phase = "health_check"
                logger.info(f"Checking {provider} health before model: {model_name}")
                provider_ready = await self._wait_for_provider_ready(provider, max_wait=60)
                if not provider_ready:
                    error_msg = f"{provider} nicht erreichbar - ueberspringe {model_name}"
                    logger.warning(f"{model_name} SKIPPED: provider not reachable")
                    compare_state.results.append({
                        "model": model_name, "text": "", "chars": 0,
                        "duration_seconds": 0, "pages_processed": 0, "error": error_msg,
                    })
                    continue

                model_params = self.get_model_params(model_name)
                optimal_image_size = model_params["max_image_size"]
                render_dpi = model_params.get("render_dpi", 200)

                compare_state.phase = "model_loading"
                logger.info(f"Testing model: {model_name} (image: {optimal_image_size}px, DPI: {render_dpi}, ctx: {model_params['num_ctx']}, repeat_pen: {model_params['repeat_penalty']})")

                if is_pdf and render_dpi not in dpi_image_cache:
                    compare_state.phase = "convert"
                    logger.info(f"Rendering PDF at {render_dpi} DPI for {model_name}")
                    dpi_images = await asyncio.get_running_loop().run_in_executor(
                        None, lambda dpi=render_dpi: convert_from_bytes(file_bytes, dpi=dpi)
                    )
                    dpi_image_cache[render_dpi] = dpi_images

                source_images = dpi_image_cache.get(render_dpi, preview_images) if is_pdf else preview_images
                pages_to_process = [(idx, source_images[idx]) for idx in page_indices]

                prepared_pages = []
                for idx, img in pages_to_process:
                    src_w, src_h = img.size
                    prepared_bytes = self._prepare_image(img, max_size=optimal_image_size)
                    debug_img = Image.open(io.BytesIO(prepared_bytes))
                    prep_w, prep_h = debug_img.size
                    logger.debug(f"{model_name} page {idx+1}: source={src_w}x{src_h}, prepared={prep_w}x{prep_h}, bytes={len(prepared_bytes)}, format={debug_img.format}")
                    prepared_pages.append((idx, prepared_bytes))

                model_start = time.time()
                page_texts = []
                error_msg = None

                try:
                    for page_idx, prepared_bytes in prepared_pages:
                        compare_state.phase = "ocr_page"
                        compare_state.current_page = page_idx + 1
                        compare_state.elapsed_seconds = round(time.time() - job_start, 1)

                        page_text = await self._ocr_single_image(
                            prepared_bytes, model_name, provider,
                            page_num=page_idx + 1, total_pages=total_pages, timeout=300.0,
                        )
                        preview = page_text[:200].replace('\n', ' ') if page_text else "(empty)"
                        logger.debug(f"{model_name} page {page_idx+1} result: {len(page_text or '')} chars, preview: {preview}")
                        page_texts.append(page_text)
                except Exception as e:
                    error_msg = str(e)
                    error_type = type(e).__name__
                    logger.error(f"[Compare] Model {model_name} failed ({error_type}): {e}")
                    logger.error(f"{model_name} FAILED ({error_type}): {e}")

                model_duration = time.time() - model_start
                full_text = "\n\n".join(page_texts) if page_texts else ""

                compare_state.results.append({
                    "model": model_name,
                    "text": full_text,
                    "chars": len(full_text),
                    "duration_seconds": round(model_duration, 2),
                    "pages_processed": len(page_texts),
                    "error": error_msg,
                })
                logger.info(f"{model_name}: {len(full_text)} chars in {model_duration:.1f}s")

                compare_state.phase = "unloading"
                compare_state.elapsed_seconds = round(time.time() - job_start, 1)
                await self._unload_model_from_vram(model_name)

                if error_msg:
                    logger.error("Modell hatte Fehler, warte 5s auf Recovery...")
                    await asyncio.sleep(5)

            compare_state.phase = "done"
            compare_state.elapsed_seconds = round(time.time() - job_start, 1)
            logger.info(f"All {len(slots)} models done in {compare_state.elapsed_seconds}s")

        except Exception as e:
            compare_state.phase = "error"
            compare_state.error = str(e)
            compare_state.elapsed_seconds = round(time.time() - job_start, 1)
            logger.error(f"[Compare] Job failed: {e}")
        finally:
            compare_state.running = False

    async def processor_loop(self, paperless_client):
        assert self.state is not None
        """Continuous background loop to check for new documents."""
        from datetime import datetime

        logger.info("Processor started")
        logger.info("Processor started")

        _ocrfinish_tag = None
        _ocrpruefen_tag = None
        _ocrerror_tag = None

        async def _get_exclude_tag_ids():
            assert self.state is not None
            nonlocal _ocrfinish_tag, _ocrpruefen_tag, _ocrerror_tag
            try:
                if _ocrfinish_tag is None:
                    _ocrfinish_tag = await paperless_client.get_or_create_tag(TAG_OCR_FINISH)
                if _ocrpruefen_tag is None:
                    _ocrpruefen_tag = await paperless_client.get_or_create_tag(TAG_OCR_REVIEW)
                if _ocrerror_tag is None:
                    _ocrerror_tag = await paperless_client.get_or_create_tag(TAG_OCR_ERROR)
                return [
                    t["id"] for t in [_ocrfinish_tag, _ocrpruefen_tag, _ocrerror_tag]
                    if t and t.get("id")
                ]
            except Exception:
                return []

        while self.state.processor.enabled:
            try:
                self.state.processor.running = True

                if self.state.batch.running:
                    logger.info("Processor: Batch aktiv, ueberspringe diesen Zyklus")
                else:
                    logger.info("Processor checking for new documents...")
                    logger.info(f"Processor check at {datetime.now().isoformat()}")

                    should_run = True
                    try:
                        exclude_ids = await _get_exclude_tag_ids()
                        if exclude_ids:
                            pending_count = await paperless_client.get_document_count(
                                tags_id_none=exclude_ids
                            )
                            if pending_count == 0:
                                logger.info("Processor: Keine neuen Dokumente – überspringe diesen Zyklus")
                                should_run = False
                            else:
                                logger.info(f"Processor: ~{pending_count} Dokument(e) ohne OCR gefunden, starte Batch...")
                    except Exception as check_err:
                        logger.warning(f"Processor: Pre-Check fehlgeschlagen, starte Batch trotzdem: {check_err}")

                    if should_run:
                        await self.batch_ocr(
                            paperless_client,
                            mode="all",
                            set_finish_tag=True,
                            remove_runocr_tag=True
                        )

                self.state.processor.last_run = datetime.now().isoformat()

            except Exception as e:
                logger.error(f"Processor error: {e}")
                logger.info(f"Processor error: {e}")

            self.state.processor.running = False
            interval_min = self.state.processor.interval_minutes or 1
            for _ in range(interval_min * 60):
                if not self.state.processor.enabled:
                    break
                await asyncio.sleep(1)

        self.state.processor.running = False
        logger.info("Processor stopped")
        logger.info("Processor stopped")

    # ── Extracted router business logic ─────────────────────────────────────

    def get_ocr_settings_with_state(self) -> dict:
        """Get OCR settings defaults merged with processor state."""
        settings = {
            "model": DEFAULT_OCR_MODEL,
            "max_image_size": 1344,
            "smart_skip_enabled": True,
        }
        settings["processor_enabled"] = self.state.processor.enabled
        settings["processor_interval"] = self.state.processor.interval_minutes
        return settings

    async def save_ocr_settings(
        self, model: str, max_image_size: int, smart_skip_enabled: bool, config_svc: Any,
    ) -> dict:
        """Save OCR settings to KV store."""
        await config_svc.set("ocr_model", model, "str")
        await config_svc.set("max_image_size", str(max_image_size), "int")
        await config_svc.set("smart_skip_enabled", str(smart_skip_enabled).lower(), "bool")
        return {
            "success": True,
            "model": model,
            "max_image_size": max_image_size,
            "smart_skip_enabled": smart_skip_enabled,
        }

    async def persist_processor_enabled(self, enabled: bool, config_svc: Any) -> None:
        """Persist processor enabled flag to AppSettings."""
        await config_svc.set(
            "ocr_processor_enabled",
            "true" if enabled else "false",
            "bool",
        )

    async def configure_processor(
        self, enabled: bool, interval_minutes: int, config_svc: Any, client: Any,
    ) -> dict:
        """Configure processor: set interval, persist, start/stop loop."""
        self.state.processor.interval_minutes = max(1, interval_minutes)
        try:
            await self.persist_processor_enabled(enabled, config_svc)
        except Exception as e:
            logger.warning(f"Could not persist ocr_processor_enabled to KV: {e}")

        if enabled and not self.state.processor.enabled:
            self.state.processor.enabled = True
            loop = asyncio.get_running_loop()
            self.state.processor.task = loop.create_task(self.processor_loop(client))
        elif not enabled and self.state.processor.enabled:
            self.state.processor.enabled = False

        return self.get_processor_status_dict()

    def get_processor_status_dict(self) -> dict:
        """Return processor status as dict."""
        return {
            "enabled": self.state.processor.enabled,
            "running": self.state.processor.running,
            "interval_minutes": self.state.processor.interval_minutes,
            "last_run": self.state.processor.last_run,
        }

    def pause_batch(self) -> dict:
        """Pause the running batch OCR job."""
        if not self.state.batch.running:
            return {"success": False, "message": "Kein Batch-Job aktiv"}
        self.state.batch.paused = True
        return {"success": True, "message": "Batch-Job pausiert", "paused": True}

    def resume_batch(self) -> dict:
        """Resume the paused batch OCR job."""
        if not self.state.batch.running:
            return {"success": False, "message": "Kein Batch-Job aktiv"}
        self.state.batch.paused = False
        return {"success": True, "message": "Batch-Job fortgesetzt", "paused": False}

    def stop_batch(self) -> dict:
        """Stop the running batch OCR job."""
        if not self.state.batch.running:
            return {"stopped": False, "message": "Kein Batch-Job aktiv"}
        self.state.batch.should_stop = True
        return {"stopped": True, "message": "Batch-Job wird gestoppt..."}

    async def ocr_single_document_safe(
        self, client: Any, document_id: int, force: bool, db_session: Any,
    ) -> dict:
        """OCR a single document with lock management and error wrapping."""
        try:
            await self.state.acquire_lock("single")
            return await self.ocr_document(client, document_id, force=force, db_session=db_session)
        finally:
            self.state.release_lock()
            self.state.page_progress.pop(document_id, None)

    def get_progress_dict(self, document_id: int) -> dict:
        """Get live page-level progress for an ongoing OCR job."""
        progress = self.state.page_progress.get(document_id)
        if not progress:
            return {"active": False, "document_id": document_id}
        elapsed = time.time() - (progress.get("started_at") or time.time())
        return {
            "active": True,
            "document_id": document_id,
            "status": progress.get("status", "unknown"),
            "total_pages": progress.get("total_pages", 0),
            "done": progress.get("done", 0),
            "errors": progress.get("errors", 0),
            "current_page": progress.get("current_page", 0),
            "elapsed_seconds": round(elapsed, 1),
            "pages": progress.get("pages", []),
        }

    async def apply_ocr_result_background(
        self, client: Any, document_id: int, content: str, set_finish_tag: bool,
    ) -> None:
        """Apply OCR result in background (fire-and-forget)."""
        try:
            await self.apply_ocr_result(client, document_id, content, set_finish_tag)
            logger.info("OCR result applied successfully", extra={"document_id": document_id})
        except Exception as e:
            logger.error("OCR result apply failed", extra={"document_id": document_id, "error": str(e)})
            logger.error(f"Background apply error: {e}")

    def check_batch_not_running(self) -> None:
        """Raise if batch is already running."""
        if self.state.batch.running:
            raise ValueError("already_running")

    def get_batch_status_dict(self, llm_service: Any = None) -> dict:
        """Get current batch status including page-level progress."""
        current_doc = self.state.batch.current_document
        current_doc_id = current_doc.get("id") if isinstance(current_doc, dict) else None

        page_progress = None
        if current_doc_id and current_doc_id in self.state.page_progress:
            pp = self.state.page_progress[current_doc_id]
            page_progress = {
                "document_id": current_doc_id,
                "total_pages": pp.get("total_pages", 0),
                "done": pp.get("done", 0),
                "errors": pp.get("errors", 0),
                "current_page": pp.get("current_page", 0),
                "status": pp.get("status", "unknown"),
                "pages": pp.get("pages", []),
            }

        lock_status = llm_service.get_lock_status() if llm_service else {}
        waiting = next(
            (p for p, s in lock_status.items() if s["locked"]), None,
        ) if not self.state.batch.running else None

        return {
            "running": self.state.batch.running,
            "total": self.state.batch.total,
            "processed": self.state.batch.processed,
            "current_document": current_doc,
            "current_page_progress": page_progress,
            "errors_count": len(self.state.batch.errors),
            "log": self.state.batch.log[-50:],
            "mode": self.state.batch.mode,
            "paused": self.state.batch.paused,
            "waiting_for": waiting,
        }

    def dismiss_review_item_from_queue(self, document_id: int) -> dict:
        """Dismiss a review queue item."""
        queue = load_review_queue()
        new_queue = [q for q in queue if q["document_id"] != document_id]
        if len(new_queue) == len(queue):
            raise ValueError("Dokument nicht in Review Queue")
        save_review_queue(new_queue)
        return {"dismissed": True, "document_id": document_id}

    def ignore_review_item_permanently(self, document_id: int) -> dict:
        """Ignore document permanently: remove from review queue and add to OCR ignore list."""
        queue = load_review_queue()
        item = next((q for q in queue if q["document_id"] == document_id), None)
        title = item["title"] if item else f"Dokument {document_id}"
        new_queue = [q for q in queue if q["document_id"] != document_id]
        save_review_queue(new_queue)

        ignore_list = load_ocr_ignore_list()
        if not any(entry["document_id"] == document_id for entry in ignore_list):
            ignore_list.append({
                "document_id": document_id,
                "title": title,
                "reason": "Original besser als OCR",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            save_ocr_ignore_list(ignore_list)

        return {"ignored": True, "document_id": document_id, "title": title}

    def get_error_list_with_counts(self) -> dict:
        """Get error list with pending error counts."""
        error_list = load_ocr_error_list()
        error_counts = load_ocr_error_counts()
        return {"items": error_list, "count": len(error_list), "pending_errors": error_counts}

    def clear_all_errors(self) -> dict:
        """Clear the entire error list and error counts."""
        save_ocr_error_list([])
        save_ocr_error_counts({})
        return {"cleared": True}

    def remove_from_ocr_ignore_list(self, document_id: int) -> None:
        """Remove a document from the OCR ignore list. Raises if not found."""
        ignore_list = load_ocr_ignore_list()
        new_list = [entry for entry in ignore_list if entry["document_id"] != document_id]
        if len(new_list) == len(ignore_list):
            raise ValueError("Dokument nicht in der Ignore-Liste")
        save_ocr_ignore_list(new_list)

    async def get_preview_response(self, client: Any, document_id: int):
        """Get document preview with auto-detected media type."""
        from fastapi.responses import Response
        file_bytes = await client.get_document_preview_image(document_id)

        if file_bytes[:4] == b'%PDF':
            media_type = "application/pdf"
        elif file_bytes[:4] == b'\x89PNG':
            media_type = "image/png"
        elif file_bytes[:2] == b'\xff\xd8':
            media_type = "image/jpeg"
        elif file_bytes[:4] == b'RIFF':
            media_type = "image/webp"
        else:
            media_type = "application/pdf"
        return Response(
            content=file_bytes,
            media_type=media_type,
            headers={"Content-Disposition": "inline", "X-Content-Type-Options": "nosniff"},
        )

    async def get_thumbnail_response(self, client: Any, document_id: int):
        """Get document thumbnail with auto-detected media type."""
        from fastapi.responses import Response
        image_bytes = await client.get_document_thumbnail_bytes(document_id)
        if image_bytes[:4] == b'\x89PNG':
            return Response(content=image_bytes, media_type="image/png")
        return Response(content=image_bytes, media_type="image/webp")

    async def validate_and_start_compare(
        self, client: Any, document_id: int, slots: list, page: int, compare_state: Any,
    ) -> dict:
        """Validate compare request and start background job."""
        if compare_state.running:
            raise ValueError("already_running")

        if not slots or len(slots) == 0:
            raise ValueError("Mindestens ein Modell auswählen")
        if len(slots) > 5:
            raise ValueError("Maximal 5 Modelle gleichzeitig")

        compare_state.reset()
        compare_state.running = True
        compare_state.document_id = document_id
        compare_state.models = [s.model for s in slots]
        compare_state.total_models = len(slots)
        compare_state.phase = "starting"

        asyncio.create_task(
            self.run_compare_job(client, document_id, slots, page, compare_state)
        )
        return {"started": True, "models": len(slots)}

    def get_compare_status_dict(self, compare_state: Any) -> dict:
        """Get compare status as dict."""
        return {
            "running": compare_state.running,
            "phase": compare_state.phase,
            "current_model": compare_state.current_model,
            "current_model_index": compare_state.current_model_index,
            "total_models": compare_state.total_models,
            "current_page": compare_state.current_page,
            "total_pages": compare_state.total_pages,
            "models": compare_state.models,
            "document_id": compare_state.document_id,
            "title": compare_state.title,
            "old_content": compare_state.old_content,
            "compared_page": compare_state.compared_page,
            "results": compare_state.results,
            "error": compare_state.error,
            "elapsed_seconds": compare_state.elapsed_seconds,
        }
