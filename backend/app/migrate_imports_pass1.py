#!/usr/bin/env python3
"""Simple service imports: *_service.py → */service.py."""
import os
import re

RENAMES = [
    (r'from app\.services\.auth_service import', 'from app.services.auth.service import'),
    (r'from app\.services\.paperless_client import', 'from app.services.paperless.service import'),
    (r'from app\.services\.merge import', 'from app.services.merge.service import'),
    (r'from app\.services\.similarity import', 'from app.services.similarity.service import'),
    (r'from app\.services\.cloud_import_service import', 'from app.services.cloud_import.service import'),
    (r'from app\.services\.duplicate_service import', 'from app.services.duplicate.service import'),
    (r'from app\.services\.ocr_service import', 'from app.services.ocr.service import'),
    (r'from app\.services\.llm_service import', 'from app.services.llm.service import'),
    (r'from app\.services\.ollama_lock import', 'from app.services.llm.lock import'),
    (r'from app\.services\.statistics import', 'from app.services.statistics.service import'),
]

def migrate_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    for old, new in RENAMES:
        content = re.sub(old, new, content)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    backend_dir = "backend/app"
    modified = []

    for root, dirs, files in os.walk(backend_dir):
        dirs[:] = [d for d in dirs if d != '__pycache__']

        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                if migrate_file(filepath):
                    modified.append(filepath)

    print(f"Pass 1: Modified {len(modified)} files")
    for f in sorted(modified):
        print(f"  {f}")

if __name__ == "__main__":
    main()