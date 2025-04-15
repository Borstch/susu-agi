from langchain_community.document_loaders import SeleniumURLLoader
from langchain_core.documents import Document
from langchain_core.tools import tool


def _webpage_to_string(webpage: Document) -> str:
    return f"snippet: {webpage.page_content}, title: {webpage.metadata['title']}"


@tool(parse_docstring=True)
def read_url(urls: list[str]) -> str:
    """Extracts the content of web pages via URL.

    Args:
        urls: list of URLs of the web pages.
    """
    loader = SeleniumURLLoader(urls=urls)
    return "\n\n\n, ".join([_webpage_to_string(webpage) for webpage in loader.load()])
