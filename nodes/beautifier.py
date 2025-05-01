from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from state import AGIState


class ToolCallBeautifier:
    def __init__(self):
        self._model = self._build_prompt_template() | ChatOpenAI(
            model=self._openai_model, base_url=self._openai_base_url
        )

    def __call__(self, state: AGIState) -> AGIState | None:
        last_message = state["messages"][-1]
        if len(last_message.tool_calls) != 0:  # type: ignore
            tool_call = last_message.tool_calls[-1]  # type: ignore
            last_message.content = self._model.invoke(dict(tool_call)).content

            return {"messages": [last_message]}  # type: ignore

    @classmethod
    def _build_prompt_template(cls) -> ChatPromptTemplate:
        assert (
            cls._prompt_path.exists()
        ), f"Can not load prompt from path: {cls._prompt_path}."

        with open(cls._prompt_path) as src:
            prompt = src.read()

        return ChatPromptTemplate.from_messages(
            [("system", prompt), ("placeholder", "{messages}")]
        )

    _prompt_path = Path("prompts/tool_call_beautifier.txt")

    _openai_base_url = "https://api.proxyapi.ru/openai/v1"
    _openai_model = "gpt-3.5-turbo"
