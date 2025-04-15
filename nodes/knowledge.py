from state import AGIState
from store import get_vectore_store


class KnowledgeRetriever:
    def __init__(self) -> None:
        self._store = get_vectore_store()

    def __call__(self, state: AGIState) -> AGIState:
        last_message = state["messages"][-1]
        assert last_message.type == "human" and isinstance(
            last_message.content, str
        ), f"Got unexpected message before context retrieval:\n\n{last_message}"

        knowledge = [
            document.page_content
            for document in self._store.similarity_search(
                query=last_message.content,
                k=self._N_TOP_MATCHES,
                filter={"relevant": True},
            )
        ]

        return {"knowledge": "<knowledge>\n" + "\n".join(knowledge) + "\n</knowledge>"}

    _N_TOP_MATCHES = 5


class KnowledgeSaver:
    def __init__(self):
        self._store = get_vectore_store()

    def __call__(self, state: AGIState) -> None:
        last_message = state["messages"][-1]
        assert last_message.type == "tool" and isinstance(last_message.content, str)

        self._store.add_texts(last_message.content.split("snippet: ")[1:])
