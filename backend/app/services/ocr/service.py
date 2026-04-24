"""OCR service orchestrator."""

from __future__ import annotations

import base64
import httpx
import asyncio
import json
import logging
import time
import io
from pathlib import Path
from typing import Optional, Dict, List, Any

from PIL import Image
from pdf2image import convert_from_bytes

from app.services.llm import (
    llm_completion,
)
from app.services.llm.lock import (
    acquire as ollama_acquire,
    release as ollama_release,
    is_locked as ollama_is_locked,
    current_holder as ollama_holder,
)
from app.services.llm.protocol import LLMService
from app.services.llm.service import LLMLockTimeoutError

from .state import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_OCR_MODEL,
    OcrState,
    OcrDocumentProgress,
    PageProgress,
    TAG_RUN_OCR,
    TAG_OCR_FINISH,
    TAG_OCR_REVIEW,
    TAG_OCR_ERROR,
    QUALITY_THRESHOLD,
    MAX_ERROR_COUNT,
    REVIEW_QUEUE_FILE,
    OCR_IGNORE_FILE,
    OCR_ERROR_COUNT_FILE,
    OCR_ERROR_FILE,
)

# Import file operations from local modules
from .review import load_review_queue, save_review_queue
from .error import (
    load_ocr_error_counts,
    save_ocr_error_counts,
    increment_ocr_error,
    reset_ocr_error,
    load_ocr_error_list,
    save_ocr_error_list,
    get_ocr_error_ids,
)
from .ignore import load_ocr_ignore_list, save_ocr_ignore_list, get_ocr_ignored_ids

# Raise PIL pixel limit for large PDF pages rendered at high DPI
Image.MAX_IMAGE_PIXELS = 500_000_000  # 500 megapixels (default is ~178MP)

logger = logging.getLogger(__name__)


class OcrService:
    """Service for OCR using Ollama Vision models."""

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_OCR_MODEL,
        max_image_size: int = 2048,
        state: OcrState | None = None,
        llm_service: LLMService | None = None,
    ):
        self.ollama_urls: List[str] = [s.strip() for s in ollama_url.split(",")]
        self.url_index: int = 0
        self.model = model
        self.max_image_size = max_image_size
        self._config_lock = asyncio.Lock()
        self._configured = False
        self.state: OcrState = state or OcrState()
        self.llm_service = llm_service

    def get_current_url(self) -> str:
        return self.ollama_urls[self.url_index % len(self.ollama_urls)]

    def rotate_url(self) -> None:
        self.url_index = (self.url_index + 1) % len(self.ollama_urls)

    async def find_best_server(self) -> bool:
        """Check which server responds fastest and set it as primary."""
        best_idx = 0
        best_latency = float("inf")

        for i, url in enumerate(self.ollama_urls):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    start = time.time()
                    response = await client.get(f"{url}/api/tags")
                    latency = time.time() - start
                    if response.status_code == 200:
                        if latency < best_latency:
                            best_latency = latency
                            best_idx = i
                        logger.info(f"[OCR] Server {url} responded in {latency:.2f}s")
            except Exception as e:
                logger.warning(f"[OCR] Server {url} check failed: {e}")

        if best_latency < float("inf"):
            self.url_index = best_idx
            logger.info(f"[OCR] Selected best server: {self.get_current_url()} (latency: {best_latency:.2f}s)")
            return True
        return False

    async def _ensure_config(self) -> None:
        """Ensure Ollama is configured and accessible (called once per service instance)."""
        if self._configured:
            return
        async with self._config_lock:
            if self._configured:
                return
            if not self.ollama_urls:
                raise ValueError("No Ollama URLs configured")
            for url in self.ollama_urls:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get(f"{url}/api/tags")
                        if response.status_code == 200:
                            self._configured = True
                            logger.info(f"[OCR] Ollama connected at {url}")
                            return
                except Exception:
                    pass
            raise ConnectionError(f"Kein Ollama-Server erreichbar unter {self.ollama_urls}")

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
    def _prepare_image_for_ollama(image: Image.Image, max_size: int = 1280) -> bytes:
        """Resize image to max dimension and convert to JPEG bytes for Ollama.

        Reduces image size to save tokens and speed up OCR without losing text legibility.
        """
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)
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

    def _build_ocr_prompt(self, page_num: int = 0, total_pages: int = 0) -> str:
        """Build model-specific OCR prompt.

        Based on paperless-gpt's proven universal prompt as baseline.
        Model-specific adjustments only where absolutely needed:
        - deepseek-ocr: Minimal prompt (echoes anything longer)
        - glm-ocr: Keyword format per official docs
        - gemma3: Shorter version (echoes long prompts)
        - minicpm-v / qwen (default): Full paperless-gpt style prompt
        """
        name = (self.model or "").lower()

        # deepseek-ocr: ultra-minimal, NO <|grounding|> (that's for bounding boxes!)
        if "deepseek-ocr" in name:
            return "OCR this document."

        # glm-ocr: keyword-based per official Ollama docs
        if "glm-ocr" in name or "glm_ocr" in name:
            return "Text Recognition:"

        # gemma3: shorter prompt (echoes/repeats long prompts verbatim)
        if "gemma3" in name or "gemma-3" in name:
            prompt = (
                "Just transcribe the text in this image. Preserve the formatting and layout. "
                "Be thorough, continue until the bottom of the page. "
                "Use markdown format but without a code block."
            )
            if page_num > 0 and total_pages > 0:
                prompt += f" This is page {page_num} of {total_pages}."
            return prompt

        # Default: paperless-gpt proven prompt + German hints
        parts = [
            "Transcribe ALL text in this image EXACTLY as it appears – high quality OCR.",
            "CRITICAL: Do NOT summarize, skip, or abbreviate any content. Continue until the very bottom of the page.",
            "CRITICAL: Every single number, amount, percentage, account number, and code MUST be transcribed exactly.",
            "For tables: transcribe each row completely, including all columns and values.",
            "For checkboxes/tick boxes: write [ ] for unchecked and [X] for checked, followed by the label text.",
            "For form fields: write the label followed by the filled-in value or a blank line if empty.",
            "For structured forms (tax notices, invoices, bank statements): preserve every field label and its value.",
            "Use markdown format without code blocks. Preserve the original layout as closely as possible.",
        ]
        if page_num > 0 and total_pages > 0:
            parts.append(f"This is page {page_num} of {total_pages}.")
        parts.append(
            "The document is in German. "
            "Pay special attention to: names, dates (DD.MM.YYYY), IBANs, BIC codes, "
            "tax IDs (Steuernummer), amounts in EUR, account numbers, reference numbers, "
            "and addresses. Transcribe every value exactly as printed – no rounding, no omitting."
        )
        return "\n".join(parts)

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
                    print(f"[OCR] Instruction echo detected at line {run_start}, truncating")
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
            print(f"[OCR] Repetition cleanup: {original_lines} -> {cleaned_lines} lines (removed {original_lines - cleaned_lines} repeated lines)")

        return '\n'.join(cleaned)

    async def _ocr_single_image(self, image_bytes: bytes, page_num: int = 0, total_pages: int = 0, timeout: float = 300.0) -> str:
        """Run OCR on a single prepared image bytes block.

        Uses model-specific parameters from get_model_params().
        If a repetition loop is detected, retries with anti-loop parameters.
        """
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt_text = self._build_ocr_prompt(page_num, total_pages)
        model_params = self.get_model_params(self.model)

        # First attempt with standard parameters
        text = await self._run_ollama_ocr(image_b64, prompt_text, model_params, timeout)

        if not text:
            return None

        # Detect repetition loop: if >60% of raw output was removed
        cleaned = text["_cleaned"] if isinstance(text, dict) else text
        loop_ratio = text.get("_loop_ratio", 0) if isinstance(text, dict) else 0

        if loop_ratio > 0.6:
            print(f"[OCR] Loop detected ({loop_ratio:.0%} wasted). Retrying with anti-table prompt...")
            retry_params = {**model_params}
            retry_params["num_predict"] = min(model_params["num_predict"], 4096)

            anti_table_prompt = (
                "Transcribe ALL text in this image completely from top to bottom. "
                "Do NOT use table formatting, pipes |, or dashes ---. "
                "Write each piece of information on its own line, using colons for labels. "
                "Include every single line of text: headers, items, prices, totals, footer, company details, IBAN."
            )

            retry_text = await self._run_ollama_ocr(image_b64, anti_table_prompt, retry_params, timeout)
            if retry_text:
                retry_cleaned = retry_text["_cleaned"] if isinstance(retry_text, dict) else retry_text
                retry_ratio = retry_text.get("_loop_ratio", 0) if isinstance(retry_text, dict) else 0

                if len(retry_cleaned) > len(cleaned):
                    print(f"[OCR] Anti-table retry improved: {len(cleaned)} -> {len(retry_cleaned)} chars (loop: {retry_ratio:.0%})")
                    return retry_cleaned
                else:
                    print(f"[OCR] Retry not better ({len(retry_cleaned)} vs {len(cleaned)} chars), keeping original")

        return cleaned

    async def _run_ollama_ocr(self, image_b64: str, prompt_text: str, model_params: dict, timeout: float) -> dict | str | None:
        """Execute a single Ollama OCR request via LiteLLM."""
        name_lower = (self.model or "").lower()
        use_think_param = "qwen3" in name_lower

        print(f"[OCR][DEBUG] Model: {self.model}, repeat_pen={model_params['repeat_penalty']}, predict={model_params['num_predict']}")

        attempts = len(self.ollama_urls)
        last_error = None

        for _ in range(attempts):
            url = self.get_current_url()
            try:
                system_msg = (
                    "You are a precise OCR module. Output ONLY the verbatim transcribed text from the image – nothing else. "
                    "No summaries, no descriptions, no commentary, no 'Let me...', no 'Here is...'. "
                    "Every number, every EUR amount, every date, every code must appear EXACTLY as printed. "
                    "For tables: every row, every column, every cell value. "
                    "Missing a single number is a critical OCR failure. Raw verbatim transcription only."
                )

                # LiteLLM multimodal format (OpenAI-compatible, works with Ollama vision)
                user_content = [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]

                extra_body: dict = {
                    "options": {
                        "temperature":    model_params["temperature"],
                        "repeat_penalty": model_params["repeat_penalty"],
                        "num_ctx":        model_params["num_ctx"],
                        "num_predict":    model_params["num_predict"],
                    },
                    "keep_alive": "30m",
                }
                if use_think_param:
                    extra_body["think"] = False

                response = await llm_completion(
                    model=self.model,
                    provider="ollama",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": user_content},
                    ],
                    api_base=url,
                    extra_body=extra_body,
                    timeout=timeout,
                )

                text_content = (response.choices[0].message.content or "").strip()
                text_content = self._strip_reasoning(text_content)
                text_content = self._strip_ocr_commentary(text_content)

                raw_len = len(text_content) if text_content else 0
                if text_content:
                    text_content = self._clean_repetitions(text_content)
                cleaned_len = len(text_content) if text_content else 0
                loop_ratio = 1 - (cleaned_len / raw_len) if raw_len > 0 else 0

                if raw_len != cleaned_len:
                    print(f"[OCR] Repetition cleanup: {raw_len} -> {cleaned_len} chars ({loop_ratio:.0%} removed)")

                eval_count = 0
                if response.usage:
                    eval_count = response.usage.completion_tokens or 0
                if eval_count >= 8000:
                    print(f"[OCR] WARNING: Token limit likely hit ({eval_count} tokens)")

                return {
                    "_cleaned": text_content,
                    "_raw":     text_content,
                    "_loop_ratio": loop_ratio,
                }

            except Exception as e:
                last_error = e
                print(f"[OCR] Error on {url}: {e}")
                self.rotate_url()

        print(f"[OCR] All URLs failed. Last error: {last_error}")
        return None

    def save_stats(self, doc_id: int, duration: float, pages: int, chars: int, success: bool = True):
        """Save OCR statistics to JSON file."""
        from datetime import datetime

        stats_file = Path("/app/data/ocr_stats.json")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "duration": round(duration, 2),
            "pages": pages,
            "chars": chars,
            "model": self.model,
            "server": self.get_current_url(),
            "success": success
        }

        try:
            stats = []
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    try:
                        stats = json.load(f)
                    except:
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
            except:
                pass
        return []

    def _extract_text_from_pdf(self, file_bytes: bytes) -> Optional[str]:
        """Extract text from PDF bytes using pypdf.

        Refined Logic (v1.1.7):
        - If text is found, check metadata.
        - If metadata indicates previous OCR (Abbyy, Tesseract, Paperless), return None (force new Vision OCR).
        - If metadata indicates 'Digital Born' (Word, LaTeX, Invoice Systems), return the text (Skip OCR).
        """
        if not self.smart_skip_enabled:
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
        """OCR a document with page-level persistence. Supports resume after failures."""
        await self._ensure_config()
        start_time = time.time()
        print(f"[OCR] Starting OCR for document {document_id}")

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
            print(f"[OCR] Downloaded {len(file_bytes)} bytes ({file_mb:.1f} MB)")

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
            raise ValueError(f"Download fehlgeschlagen: {e}")

        if not force:
            native_text = self._extract_text_from_pdf(file_bytes)
            if native_text and len(native_text) > 50:
                logger.info(f"Found native text in PDF ({len(native_text)} chars). Skipping OCR.")
                print(f"[OCR] Native text found ({len(native_text)} chars). Skipping vision OCR.")
                self.state.page_progress.pop(document_id, None)
                duration = time.time() - start_time
                return {
                    "document_id": document_id, "title": title,
                    "old_content": old_content, "new_content": native_text,
                    "old_length": len(old_content), "new_length": len(native_text),
                    "ocr_duration": duration, "ocr_pages": 0, "source": "native_pdf",
                }

        self.state.page_progress[document_id].status = "converting"
        images = await self._convert_to_images(file_bytes, doc, document_id, title)
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
                print(f"[OCR] Force mode: cleared DB page cache for doc {document_id}, starting fresh")
            else:
                completed_pages = await self._load_completed_pages(db_session, document_id, total_pages)
                if completed_pages:
                    print(f"[OCR] Resume: found {len(completed_pages)} completed pages in DB")
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
                print(f"[OCR] Page {page_num}/{total_pages}: resumed from DB ({len(completed_pages[page_num])} chars)")
                continue

            self.state.page_progress[document_id].current_page = page_num
            self.state.page_progress[document_id].pages[i].status = "processing"

            model_params = self.get_model_params(self.model)
            optimal_size = max(self.max_image_size, model_params["max_image_size"])
            prepared_bytes = self._prepare_image_for_ollama(img, max_size=optimal_size)

            page_text = None
            last_error = None

            for attempt in range(1, MAX_PAGE_RETRIES + 1):
                try:
                    page_start = time.time()
                    msg = f"Processing page {page_num}/{total_pages} (attempt {attempt})"
                    logger.info(msg)
                    print(f"[OCR] {msg}")

                    page_text = await self._ocr_single_image(
                        prepared_bytes, page_num=page_num, total_pages=total_pages
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
                    print(f"[OCR] Page {page_num}: OK ({len(page_text)} chars, {page_duration:.1f}s)")
                    break

                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"OCR page {page_num} attempt {attempt} failed: {e}")
                    print(f"[OCR] Page {page_num} attempt {attempt} failed: {e}")

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
                print(f"[OCR] Page {page_num}: FAILED after {MAX_PAGE_RETRIES} attempts")

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

    async def _convert_to_images(self, file_bytes: bytes, doc: dict, document_id: int, title: str) -> list:
        """Convert file to list of PIL images."""
        model_params = self.get_model_params(self.model)
        render_dpi = model_params.get("render_dpi", 200)
        is_pdf = file_bytes[:4] == b'%PDF'
        mime_type = doc.get("mime_type", "unknown")

        if is_pdf:
            for attempt in range(2):
                try:
                    loop = asyncio.get_running_loop()
                    dpi = render_dpi if attempt == 0 else max(100, render_dpi - 50)
                    print(f"[OCR] Converting PDF at {dpi} DPI (attempt {attempt+1})…")
                    images = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda d=dpi: convert_from_bytes(file_bytes, dpi=d)
                        ),
                        timeout=300
                    )
                    print(f"[OCR] Converted PDF to {len(images)} pages at {dpi} DPI")
                    return images
                except asyncio.TimeoutError:
                    print(f"[OCR] PDF conversion timed out after 5 min (attempt {attempt+1})")
                    if attempt == 0:
                        logger.warning(f"PDF conversion timeout for doc {document_id}, retrying at lower DPI")
                        continue
                    raise ValueError(f"PDF-Konvertierung nach 10 Minuten abgebrochen – Dokument übersprungen.")
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
                        print(f"[OCR] PDF conversion failed (attempt 1), retrying: {e}")
                        await asyncio.sleep(2)
                    else:
                        raise ValueError(f"PDF-Konvertierung fehlgeschlagen nach 2 Versuchen ({mime_type}): {e}")
        else:
            try:
                img = Image.open(io.BytesIO(file_bytes))
                img.load()
                print(f"[OCR] Loaded as single image ({img.format}, {img.size[0]}x{img.size[1]})")
                return [img]
            except Exception as e:
                raise ValueError(f"Dateiformat nicht unterstützt ({mime_type}, {len(file_bytes)} bytes): {e}")
        return []

    async def _load_completed_pages(self, db_session, document_id: int, total_pages: int) -> Dict[int, str]:
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
        duration: float = 0, error_message: str = None,
    ):
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
        """Remove page results from DB after successful completion."""
        from sqlalchemy import delete
        from app.models.ocr import OcrPageResult
        try:
            await db_session.execute(
                delete(OcrPageResult).where(OcrPageResult.document_id == document_id)
            )
            await db_session.commit()
            print(f"[OCR] Cleaned up page cache for document {document_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup page results: {e}")

    async def ocr_image(self, image_bytes: bytes) -> str:
        """Legacy method for backward compat or single image bytes."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            prepared = self._prepare_image_for_ollama(img)
            return await self._ocr_single_image(prepared)
        except Exception as e:
            logger.error(f"Legacy ocr_image failed: {e}")
            raise

    async def apply_ocr_result(
        self,
        paperless_client,
        document_id: int,
        new_content: str,
        set_finish_tag: bool = True
    ) -> Dict[str, Any]:
        """Apply OCR result to document and optionally set ocrfinish tag."""
        start_time = time.time()

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
            self.save_stats(document_id, duration, 0, len(new_content), success=tag_success)
        except:
            pass

        return {"success": tag_success, "document_id": document_id}

    async def batch_ocr(
        self,
        paperless_client,
        mode: str = "all",
        document_ids: List[int] = None,
        set_finish_tag: bool = True,
        remove_runocr_tag: bool = True
    ) -> None:
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

        lock_acquired = False
        try:
            if ollama_is_locked():
                holder = ollama_holder()
                self.state.batch.log.append(f"⏳ Warte auf {holder} (Ollama belegt)...")
                logger.info(f"[OCR-Batch] Ollama belegt durch {holder}, warte...")
            lock_acquired = await ollama_acquire("ocr-batch", timeout=600)
            if not lock_acquired:
                self.state.batch.log.append("❌ Ollama-Lock nicht erhalten nach 10 Min – Abbruch.")
                self.state.batch.running = False
                return

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

            self.state.batch.log.append("🔎 Prüfe verfügbare Ollama-Server...")
            if await self.find_best_server():
                self.state.batch.log.append(f"🚀 Verbunden mit: {self.get_current_url()}")
            else:
                self.state.batch.log.append(f"⚠️ Warnung: Kein Server antwortet schnell. Nutze {self.get_current_url()}")

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
                                f"❌ Paperless nach 3 Versuchen nicht erreichbar – Batch abgebrochen."
                            )
                            logger.error(f"batch_ocr: get_documents failed after 3 attempts: {fetch_err}")
                            return

                documents = [
                    d for d in all_docs
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
                    print(f"[OCR] Skipped {skipped} ignored documents")

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
                            print(f"[OCR] Quality check failed for {doc_id} ({ratio}%), retrying OCR...")

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
                                    print(f"[OCR] Retry improved: {retry_len} chars (was {new_len - (retry_len - new_len)})")
                                elif retry_content:
                                    if retry_len >= new_len:
                                        new_content = retry_content
                                        new_len = retry_len
                                    ocr_duration += retry_result.get("ocr_duration", 0)
                                    self.state.batch.log.append(
                                        f"🔁 {doc_title}: Retry ähnliches Ergebnis ({retry_len} Zeichen)"
                                    )
                                    print(f"[OCR] Retry similar: {retry_len} chars")
                            except Exception as retry_err:
                                self.state.batch.log.append(
                                    f"🔁 {doc_title}: Retry fehlgeschlagen - {str(retry_err)}"
                                )
                                print(f"[OCR] Retry failed for {doc_id}: {retry_err}")

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
                                print(f"[OCR] Retry fixed quality for {doc_id}: {new_len} chars now passes threshold")

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
                                self.save_stats(doc_id, ocr_duration, ocr_pages, new_len, success=tag_success)
                            except:
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
                        self.save_stats(doc_id, 0, 0, 0, success=False)
                    except:
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
                            print(f"[OCR] Doc {doc_id} permanently marked as failed ({err_count} failures)")
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
            if lock_acquired:
                ollama_release("ocr-batch")
            self.state.batch.running = False
            self.state.batch.current_document = None

    async def watchdog_loop(self, paperless_client):
        """Continuous background loop to check for new documents."""
        from datetime import datetime

        logger.info("Watchdog started")
        print("[OCR] Watchdog started")

        _ocrfinish_tag = None
        _ocrpruefen_tag = None
        _ocrerror_tag = None

        async def _get_exclude_tag_ids():
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

        while self.state.watchdog.enabled:
            try:
                self.state.watchdog.running = True

                if self.state.batch.running or self.state.is_locked() or ollama_is_locked():
                    reason = "Batch" if self.state.batch.running else "Single-OCR" if self.state.is_locked() else f"Ollama belegt ({ollama_holder()})"
                    logger.info(f"Watchdog: {reason} aktiv, ueberspringe diesen Zyklus")
                else:
                    logger.info("Watchdog checking for new documents...")
                    print(f"[OCR] Watchdog check at {datetime.now().isoformat()}")

                    should_run = True
                    try:
                        exclude_ids = await _get_exclude_tag_ids()
                        if exclude_ids:
                            pending_count = await paperless_client.get_document_count(
                                tags_id_none=exclude_ids
                            )
                            if pending_count == 0:
                                logger.info("Watchdog: Keine neuen Dokumente – überspringe diesen Zyklus")
                                should_run = False
                            else:
                                logger.info(f"Watchdog: ~{pending_count} Dokument(e) ohne OCR gefunden, starte Batch...")
                    except Exception as check_err:
                        logger.warning(f"Watchdog: Pre-Check fehlgeschlagen, starte Batch trotzdem: {check_err}")

                    if should_run:
                        await self.batch_ocr(
                            paperless_client,
                            mode="all",
                            set_finish_tag=True,
                            remove_runocr_tag=True
                        )

                self.state.watchdog.last_run = datetime.now().isoformat()

            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                print(f"[OCR] Watchdog error: {e}")

            self.state.watchdog.running = False
            interval_min = self.state.watchdog.get("interval_minutes", 1)
            for _ in range(interval_min * 60):
                if not self.state.watchdog.enabled:
                    break
                await asyncio.sleep(1)

        self.state.watchdog.running = False
        logger.info("Watchdog stopped")
        print("[OCR] Watchdog stopped")
