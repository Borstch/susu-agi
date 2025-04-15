import numpy as np
import pandas as pd

from store import get_vectore_store

_K = 10
_questions = [
    "Какая столица Франции?",
    "Когда началась Вторая мировая война?",
    "Кто написал 'Войну и мир'?",
    "Какой химический элемент имеет символ 'O'?",
    "Сколько планет в Солнечной системы?",
    "Кто изобрел телефон?",
    "Какова формула воды?",
    "Какая самая высокая гора в мире?",
    "Кто первый человек, побывавший в космосе?",
    "Какой год был объявлен Международным годом астрономии?",
    "Кто выиграл последний чемпионат мира по футболу?",
    "Какие основные достопримечательности Рима?",
    "Какие основные функции языка программирования Python?",
    "Как работает блокчейн?",
    "Какие основные принципы квантовой механики?",
]


def get_total_relevant_documents(question_id: int) -> int:
    assert 0 < question_id
    df = pd.read_csv("scripts/knowledge.csv", header=None, skiprows=1, index_col=0)
    return sum([int(v.split("|")[0]) for v in df.values[question_id - 1, :]])


def recall_at_k(scores: pd.DataFrame, k: float) -> float:
    return np.mean(
        scores.values[:, :k].sum(axis=1)
        / np.array(
            [
                get_total_relevant_documents(question_id=score_id + 1)
                for score_id in range(len(scores))
            ]
        )
    )


def precision_at_k(scores: pd.DataFrame, k: float) -> float:
    return np.mean(scores.values[:, :k].sum(axis=1) / k)


def main():
    store = get_vectore_store()
    scores = []

    for question_id, question in enumerate(_questions, start=1):
        documents = store.similarity_search(query=question, k=_K)

        scores.append(
            [
                int(
                    document.metadata["question_id"] == question_id
                    and document.metadata["relevant"]
                )
                for document in documents
            ]
        )

    df = pd.DataFrame(scores, index=range(1, len(_questions) + 1))
    for k in (1, 3, 5, 10):
        p = precision_at_k(df, k=k)
        r = recall_at_k(df, k=k)
        print(f"Precision at {k} is: {p:.4f}")
        print(f"Recall at {k} is: {r:.4f}")
        print(f"F-Score at k is: {((2 * p * r) / (p + r)):.4f}")


if __name__ == "__main__":
    main()
