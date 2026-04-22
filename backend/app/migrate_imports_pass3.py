#!/usr/bin/env python3
"""Special case imports: auth middleware, state accessors, loop renames."""
import os
import re

# Special case transformations
TRANSFORMS = [
    # Auth middleware (SessionAuthMiddleware was from auth_service)
    (r'from app\.services\.auth\.service import SessionAuthMiddleware',
     'from app.services.auth.middleware import SessionAuthMiddleware'),

    # OCR state from ocr_service → ocr.state
    (r'from app\.services\.ocr_service import batch_state',
     'from app.services.ocr.state import batch_state'),
    (r'from app\.services\.ocr_service import watchdog_state',
     'from app.services.ocr.state import watchdog_state'),
    (r'from app\.services\.ocr_service import single_ocr_running',
     'from app.services.ocr.state import single_ocr_running'),
    (r'from app\.services\.ocr_service import ocr_page_progress',
     'from app.services.ocr.state import ocr_page_progress'),
    (r'from app\.services\.ocr_service import DEFAULT_OLLAMA_URL',
     'from app.services.ocr.state import DEFAULT_OLLAMA_URL'),
    (r'from app\.services\.ocr_service import DEFAULT_OCR_MODEL',
     'from app.services.ocr.state import DEFAULT_OCR_MODEL'),
    (r'from app\.services\.ocr_service import TAG_RUN_OCR',
     'from app.services.ocr.state import TAG_RUN_OCR'),
    (r'from app\.services\.ocr_service import TAG_OCR_FINISH',
     'from app.services.ocr.state import TAG_OCR_FINISH'),
    (r'from app\.services\.ocr_service import TAG_OCR_REVIEW',
     'from app.services.ocr.state import TAG_OCR_REVIEW'),
    (r'from app\.services\.ocr_service import TAG_OCR_ERROR',
     'from app.services.ocr.state import TAG_OCR_ERROR'),

    # OCR file operations
    (r'from app\.services\.ocr_service import load_review_queue',
     'from app.services.ocr.review import load_review_queue'),
    (r'from app\.services\.ocr_service import save_review_queue',
     'from app.services.ocr.review import save_review_queue'),
    (r'from app\.services\.ocr_service import load_ocr_ignore_list',
     'from app.services.ocr.ignore import load_ocr_ignore_list'),
    (r'from app\.services\.ocr_service import save_ocr_ignore_list',
     'from app.services.ocr.ignore import save_ocr_ignore_list'),
    (r'from app\.services\.ocr_service import get_ocr_ignored_ids',
     'from app.services.ocr.ignore import get_ocr_ignored_ids'),
    (r'from app\.services\.ocr_service import load_ocr_error_counts',
     'from app.services.ocr.error import load_ocr_error_counts'),
    (r'from app\.services\.ocr_service import save_ocr_error_counts',
     'from app.services.ocr.error import save_ocr_error_counts'),
    (r'from app\.services\.ocr_service import increment_ocr_error',
     'from app.services.ocr.error import increment_ocr_error'),
    (r'from app\.services\.ocr_service import reset_ocr_error',
     'from app.services.ocr.error import reset_ocr_error'),
    (r'from app\.services\.ocr_service import load_ocr_error_list',
     'from app.services.ocr.error import load_ocr_error_list'),
    (r'from app\.services\.ocr_service import save_ocr_error_list',
     'from app.services.ocr.error import save_ocr_error_list'),
    (r'from app\.services\.ocr_service import get_ocr_error_ids',
     'from app.services.ocr.error import get_ocr_error_ids'),

    # Cloud import state and loop
    (r'from app\.services\.cloud_import_service import get_cloud_sync_state',
     'from app.services.cloud_import.state import get_cloud_sync_state'),
    (r'from app\.services\.cloud_import_service import _cloud_sync_state',
     'from app.services.cloud_import.state import _cloud_sync_state'),
    (r'from app\.services\.cloud_import_service import cloud_sync_loop',
     'from app.services.cloud_import.sync_loop import cloud_sync_loop'),

    # Duplicate state
    (r'from app\.services\.duplicate_service import get_scan_state',
     'from app.services.duplicate.state import get_scan_state'),
    (r'from app\.services\.duplicate_service import _scan_state',
     'from app.services.duplicate.state import _scan_state'),

    # Classifier loop rename (was _auto_classify_loop in router, now auto_classify_loop in service)
    (r'from app\.routers\.classifier import _auto_classify_loop',
     'from app.services.classifier.auto_classify_loop import auto_classify_loop'),
]

def migrate_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    for old_pattern, new_replacement in TRANSFORMS:
        content = re.sub(old_pattern, new_replacement, content)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    backend_dir = "backend/app"
    modified = []

    for root, dirs, files in os.walk(backend_dir):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != '__pycache__']

        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                if migrate_file(filepath):
                    modified.append(filepath)

    print(f"Pass 3: Modified {len(modified)} files")
    for f in sorted(modified):
        print(f"  {f}")

if __name__ == "__main__":
    main()