from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools.vector_tools import load_vectorstore

QUESTIONS = [
    "Ce drepturi are persoana vizată conform GDPR?",
    "Ce obligații are operatorul la colectarea datelor personale conform GDPR?",
    "Care sunt principiile prelucrării datelor personale conform GDPR?",
    "Ce prevede Legea 98/2016 despre principiile achizițiilor publice?",
    "Ce este o clauză abuzivă potrivit ANPC?",
    "Ce drepturi are consumatorul potrivit ghidurilor ANPC?",
    "Ce condiții sunt relevante pentru forța majoră conform Codului Civil?",
    "Ce reguli generale apar în documentele UNCITRAL despre comerț electronic?",
    "Ce probleme apar la semnătura electronică potrivit UNCITRAL?",
    "Ce obligații contractuale apar în contractele publice din corpus?",
]


def main() -> None:
    load_dotenv(ROOT / ".env")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)

    vs = load_vectorstore(str(ROOT / "vectorstore"))
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    rows = []
    for q in QUESTIONS:
        docs = vs.similarity_search(q, k=10)
        contexts = [d.page_content for d in docs]
        joined_context = "\n\n".join(contexts)
        prompt = (
    "Răspunde în maximum 4 fraze, strict pe baza contextului juridic de mai jos. "
    "Folosește formulări apropiate de context. "
    "Dacă informația nu apare clar în context, spune: Nu se poate stabili din context.\n\n"
    f"CONTEXT:\n{joined_context}\n\nÎNTREBARE: {q}"
)
        answer = llm.invoke(prompt).content
        rows.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": "",
        })

    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    out = {
        "scores": result.scores,
        "summary": result.to_pandas().mean(numeric_only=True).to_dict(),
        "note": "Pragul țintă al proiectului: >= 0.6 pentru metricile RAGAS.",
    }
    (logs / "rag_evaluation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
