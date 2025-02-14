from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from state import AGIState


class ExecutionAgent:
    def __init__(self, tools: list[BaseTool]):
        model = ChatOpenAI(model=self._openai_model, base_url=self._openai_base_url)
        model_with_tools = model.bind_tools(tools)

        self._agent = self._build_prompt_template() | model_with_tools

    def __call__(self, state: AGIState) -> AGIState:
        assert "knowledge" in state
        knowledge = "<knowledge>\n" + "\n".join(state["knowledge"]) + "\n</knowledge>"

        response = self._agent.invoke(
            {
                "messages": state["messages"],
                "knowledge": knowledge,
            }
        )
        return {"messages": [response]}

    @classmethod
    def _build_prompt_template(cls) -> ChatPromptTemplate:
        assert (
            cls._system_prompt_path.exists()
        ), f"Can not load system prompt from path: {cls._system_prompt_path}."

        with open(cls._system_prompt_path) as src:
            system_prompt = src.read()

        return ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("placeholder", "{messages}")]
        )

    _system_prompt_path = Path("prompts/system/execution_agent.txt")

    _openai_base_url = "https://api.proxyapi.ru/openai/v1"
    _openai_model = "gpt-3.5-turbo"
