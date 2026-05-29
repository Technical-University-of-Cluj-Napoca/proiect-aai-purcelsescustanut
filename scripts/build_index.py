from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools.pdf_tools import load_corpus
from src.tools.vector_tools import build_index


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(ROOT / "corpus"))
    parser.add_argument("--persist", default=str(ROOT / "vectorstore"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    docs = load_corpus(args.corpus)
    print(f"Documente valide încărcate: {len(docs)}")
    if len(docs) < 15:
        print("[WARN] Cerința proiectului cere minimum 15 documente juridice.")
    build_index(docs, persist_directory=args.persist, force=args.force)
    print(f"Index ChromaDB salvat în: {args.persist}")


if __name__ == "__main__":
    main()
