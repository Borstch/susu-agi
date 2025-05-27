from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools.base import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing_extensions import Literal

from nodes.beautifier import ToolCallBeautifier
from nodes.execution_agent import ExecutionAgent
from nodes.human import QuestionBuilder, ReviewerBuilder
from nodes.knowledge import KnowledgeRetriever, KnowledgeSaver
from state import AGIState
from tools.url_reader import read_url


def _route_after_llm(
    state: AGIState,
) -> Literal["Модуль контроля диалога", "Модуль интерактивного контроля"]:
    if len(state["messages"][-1].tool_calls) == 0:  # type: ignore
        return "Модуль контроля диалога"
    else:
        return "Модуль интерактивного контроля"


class SUSUAGI:
    def __init__(self, thread_id: str):
        builder = StateGraph(AGIState)
        question_node_builder = QuestionBuilder(
            approved_path="Модуль извлечения знаний", rejected_path=END
        )
        review_node_builder = ReviewerBuilder(
            approved_path="Окружение", rejected_path="Процессор "
        )

        builder.add_node("Модуль извлечения знаний", KnowledgeRetriever())
        builder.add_node("Модуль контроля диалога", question_node_builder.build())
        builder.add_node("Процессор ", ExecutionAgent(self._tools))
        builder.add_node("Модуль вежливости ", ToolCallBeautifier(self._tools))
        builder.add_node("Окружение", ToolNode(tools=self._tools))
        builder.add_node("Модуль интерактивного контроля", review_node_builder.build())
        builder.add_node("Модуль самообучения", KnowledgeSaver())

        builder.add_edge(START, "Модуль извлечения знаний")
        builder.add_edge("Модуль извлечения знаний", "Процессор ")
        builder.add_edge("Процессор ", "Модуль вежливости ")
        builder.add_conditional_edges("Модуль вежливости ", _route_after_llm)
        builder.add_edge("Окружение", "Модуль самообучения")
        builder.add_edge("Модуль самообучения", "Процессор ")

        memory = MemorySaver()

        self._config = RunnableConfig({"configurable": {"thread_id": thread_id}})
        self._graph = builder.compile(checkpointer=memory)

    def run_untill_interrupt(
        self, input: dict[str, list[AnyMessage]] | Command
    ) -> AnyMessage:
        return self._graph.invoke(input, config=self._config)["messages"][-1]

    def get_next_node_name(self) -> str | None:
        try:
            return self._graph.get_state(self._config).next[0]
        except IndexError:
            return None

    _tools: list[BaseTool] = [DuckDuckGoSearchResults(), read_url]
