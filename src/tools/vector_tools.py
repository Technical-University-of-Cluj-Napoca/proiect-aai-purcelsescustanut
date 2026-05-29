from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _is_non_empty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def build_index(
    documents: list[dict[str, Any]],
    persist_directory: str = "vectorstore",
    collection_name: str = "legal_corpus",
    force: bool = False,
) -> Chroma:
    """Construiește indexul Chroma persistent.

    Chunking: folosesc ~500 tokeni cu 50 suprapunere, aproximat prin 2000/200
    caractere. Pentru română, un token are frecvent 3-5 caractere. Suprapunerea
    păstrează contextul când o propoziție juridică lungă cade la granița dintre
    două bucăți.
    """
    persist_path = Path(persist_directory)
    if _is_non_empty_dir(persist_path):
        if not force:
            raise RuntimeError(
                f"{persist_directory}/ există deja și nu e gol. Șterge-l sau rulează cu --force."
            )
        shutil.rmtree(persist_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )

    chunks: list[Document] = []
    for doc_idx, item in enumerate(documents):
        base_meta = dict(item["metadata"])
        for chunk_idx, chunk in enumerate(splitter.split_text(item["text"])):
            chunks.append(Document(
                page_content=chunk,
                metadata={**base_meta, "doc_id": doc_idx, "chunk_id": chunk_idx},
            ))

    if not chunks:
        raise ValueError("Nu există documente/chunk-uri valide pentru indexare.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    return vectorstore


def load_vectorstore(
    persist_directory: str = "vectorstore",
    collection_name: str = "legal_corpus",
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
