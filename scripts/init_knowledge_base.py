import pandas as pd
from langchain_core.documents import Document

from store import get_vectore_store


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
                    page_content=content,
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
