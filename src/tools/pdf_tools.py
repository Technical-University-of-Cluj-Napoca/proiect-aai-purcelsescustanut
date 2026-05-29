from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pdfplumber


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*Pagina\s+\d+\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_pdf_text(path: Path) -> tuple[str, int]:
    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
        return clean_text("\n\n".join(pages)), len(pdf.pages)


def extract_text_file(path: Path) -> tuple[str, int]:
    return clean_text(path.read_text(encoding="utf-8", errors="ignore")), 1


def infer_doc_type(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    for candidate in ["gdpr", "legi", "contracte", "uncitral", "anpc"]:
        if candidate in parts:
            return candidate
    return "unknown"


def load_corpus(corpus_dir: str | Path) -> list[dict[str, Any]]:
    """Parcurge recursiv corpus/ și returnează text + metadate.

    Sunt acceptate PDF, TXT și MD. Evit PDF-urile scanate deoarece pdfplumber
    întoarce text gol; pentru acelea ar trebui adăugat OCR separat.
    """
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directorul corpus nu există: {root}")

    documents: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        if path.suffix.lower() == ".pdf":
            text, page_count = extract_pdf_text(path)
        else:
            text, page_count = extract_text_file(path)
        if len(text) < 100:
            print(f"[WARN] Ignor document cu text prea scurt/scanat: {path}")
            continue
        documents.append({
            "text": text,
            "metadata": {
                "source": str(path.relative_to(root)),
                "title": path.stem.replace("_", " "),
                "page_count": page_count,
                "type": infer_doc_type(path),
                "file_name": path.name,
            },
        })
    return documents
