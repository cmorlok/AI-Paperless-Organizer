#!/usr/bin/env python3
"""Protocol import distribution - split multi-line protocol imports by destination."""
import os
import re

# Map of protocol names to their service sub-package
PROTOCOL_MAP = {
    'PaperlessClient': 'paperless',
    'MergeService': 'merge',
    'SimilarityService': 'similarity',
    'CloudImportService': 'cloud_import',
    'OcrService': 'ocr',
    'LLMService': 'llm',
    'DocumentClassifierService': 'classifier',
    'RAGService': 'rag',
    'DuplicateService': 'duplicate',
    'StatisticsService': 'statistics',
}

def transform_multiline_imports(content):
    """Handle multi-line imports with parentheses."""
    pattern = r'from app\.services\.protocols import\s*\((.*?)\n\)'

    def replacer(match):
        inner = match.group(1)
        imports = [line.strip().rstrip(',') for line in inner.strip().split('\n') if line.strip()]

        by_svc = {}
        for imp in imports:
            if ' as ' in imp:
                parts = imp.split(' as ')
                name = parts[0].strip()
                alias = parts[1].strip()
                if name in PROTOCOL_MAP:
                    svc = PROTOCOL_MAP[name]
                    if svc not in by_svc:
                        by_svc[svc] = []
                    by_svc[svc].append(f'{name} as {alias}')
                else:
                    if 'protocols' not in by_svc:
                        by_svc['protocols'] = []
                    by_svc['protocols'].append(imp)
            elif imp in PROTOCOL_MAP:
                svc = PROTOCOL_MAP[imp]
                if svc not in by_svc:
                    by_svc[svc] = []
                by_svc[svc].append(imp)
            else:
                if 'protocols' not in by_svc:
                    by_svc['protocols'] = []
                by_svc['protocols'].append(imp)

        new_lines = []
        for svc, imps in by_svc.items():
            if svc == 'protocols':
                new_lines.append(f'from app.services.protocols import {", ".join(imps)}')
            else:
                new_lines.append(f'from app.services.{svc}.protocol import {", ".join(imps)}')

        return '\n'.join(new_lines)

    return re.sub(pattern, replacer, content, flags=re.DOTALL)

def transform_single_import(line):
    """Transform a single-line protocol import."""
    if 'from app.services.protocols import' not in line:
        return None

    rest = line.strip()[len('from app.services.protocols import'):].strip()
    if not rest:
        return None

    rest = rest.rstrip('\\').rstrip(',')
    imports = [i.strip() for i in rest.split(',')]

    by_svc = {}
    for imp in imports:
        if ' as ' in imp:
            parts = imp.split(' as ')
            name = parts[0].strip()
            alias = parts[1].strip()
            if name in PROTOCOL_MAP:
                svc = PROTOCOL_MAP[name]
                if svc not in by_svc:
                    by_svc[svc] = []
                by_svc[svc].append(f'{name} as {alias}')
            else:
                if 'protocols' not in by_svc:
                    by_svc['protocols'] = []
                by_svc['protocols'].append(imp)
        elif imp in PROTOCOL_MAP:
            svc = PROTOCOL_MAP[imp]
            if svc not in by_svc:
                by_svc[svc] = []
            by_svc[svc].append(imp)
        else:
            if 'protocols' not in by_svc:
                by_svc['protocols'] = []
            by_svc['protocols'].append(imp)

    new_lines = []
    for svc, imps in by_svc.items():
        if svc == 'protocols':
            new_lines.append(f'from app.services.protocols import {", ".join(imps)}')
        else:
            new_lines.append(f'from app.services.{svc}.protocol import {", ".join(imps)}')

    return new_lines

def migrate_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    content = transform_multiline_imports(content)

    lines = content.split('\n')
    new_lines = []
    modified = content != original

    for line in lines:
        result = transform_single_import(line)
        if result:
            for new_line in result:
                new_lines.append(new_line)
            if len(result) > 1 or (result and result[0] != line.rstrip()):
                modified = True
        else:
            new_lines.append(line)

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))

    return modified

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

    print(f"Pass 2: Modified {len(modified)} files")
    for f in sorted(modified):
        print(f"  {f}")

if __name__ == "__main__":
    main()