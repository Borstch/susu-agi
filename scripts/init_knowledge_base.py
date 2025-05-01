from datetime import datetime, timedelta

import pandas as pd
from langchain_core.documents import Document

from store import get_vectore_store


def _get_current_date() -> str:
    return (datetime.today() - timedelta(days=10)).strftime("%Y-%m-%d")


def main():
    knowledge = pd.read_csv(
        "scripts/knowledge.csv", header=None, skiprows=1, index_col=0
    ).values

    documents = []
    for question_id, question_knowledge in enumerate(knowledge, start=1):
        for knowledge_piece in question_knowledge:
            relevance, content = knowledge_piece.split("|")
            documents.append(
                Document(
                    page_content=f"Получено {_get_current_date()}:\n\n\n" + content,
                    metadata={
                        "question_id": question_id,
                        "relevant": bool(int(relevance)),
                    },
                )
            )

    store = get_vectore_store()
    print(store.add_documents(documents))


if __name__ == "__main__":
    main()
