from os import getenv

from langchain_community.vectorstores import SQLiteVec
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings

_database_file = getenv("KNOWLEDGE_DB_PATH", "knowledge.db")


def _create_embedder() -> Embeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url="https://api.proxyapi.ru/openai/v1",
    )


def get_vectore_store() -> VectorStore:
    return SQLiteVec(
        table="knowledge",
        connection=None,
        embedding=_create_embedder(),
        db_file=_database_file,
    )
