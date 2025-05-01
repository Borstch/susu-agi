from datetime import datetime, timezone
from inspect import signature
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from state import AGIState


def _get_current_date() -> str:
    return (datetime.now(timezone.utc)).strftime("%Y-%m-%d")


def _render_text_description(tools: list[BaseTool]) -> str:
    """Render the tool name and description in plain text.

    Args:
        tools: The tools to render.

    Returns:
        The rendered text.

    Output will be in the format of:

    .. code-block:: markdown

        search: This tool is used for search
        calculator: This tool is used for math
    """
    descriptions = []
    for tool in tools:
        if hasattr(tool, "func") and tool.func:
            sig = signature(tool.func)
            description = f"{tool.name}{sig} - {tool.description}"
        else:
            description = f"{tool.name} - {tool.description}"

        descriptions.append(description)
    return "\n".join(descriptions)


class ExecutionAgent:
    def __init__(self, tools: list[BaseTool]):
        model = ChatOpenAI(model=self._openai_model, base_url=self._openai_base_url)
        model_with_tools = model.bind_tools(tools)

        self._agent = self._build_prompt_template(tools) | model_with_tools

    def __call__(self, state: AGIState) -> AGIState:
        assert "knowledge" in state

        response = self._agent.invoke(state)
        return {"messages": [response]}

    @classmethod
    def _build_prompt_template(cls, tools: list[BaseTool]) -> ChatPromptTemplate:
        assert (
            cls._system_prompt_path.exists()
        ), f"Can not load system prompt from path: {cls._system_prompt_path}."

        with open(cls._system_prompt_path) as src:
            system_prompt = src.read()

        return ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("placeholder", "{messages}")]
        ).partial(
            tools=_render_text_description(list(tools)),
            tool_names=", ".join([t.name for t in tools]),
            today=_get_current_date(),
        )

    _system_prompt_path = Path("prompts/system/execution_agent.txt")

    _openai_base_url = "https://api.proxyapi.ru/openai/v1"
    _openai_model = "gpt-3.5-turbo"
