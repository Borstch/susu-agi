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


def _route_after_llm(state: AGIState) -> Literal["next_question", "human_review"]:
    if len(state["messages"][-1].tool_calls) == 0:  # type: ignore
        return "next_question"
    else:
        return "human_review"


class SUSUAGI:
    def __init__(self, thread_id: str):
        builder = StateGraph(AGIState)
        question_node_builder = QuestionBuilder(
            approved_path="retrieval", rejected_path=END
        )
        review_node_builder = ReviewerBuilder(
            approved_path="tools", rejected_path="execution_agent"
        )

        builder.add_node("retrieval", KnowledgeRetriever())
        builder.add_node("next_question", question_node_builder.build())
        builder.add_node("execution_agent", ExecutionAgent(self._tools))
        builder.add_node("tool_call_beautifier", ToolCallBeautifier(self._tools))
        builder.add_node("tools", ToolNode(tools=self._tools))
        builder.add_node("human_review", review_node_builder.build())
        builder.add_node("knowledge_agent", KnowledgeSaver())

        builder.add_edge(START, "retrieval")
        builder.add_edge("retrieval", "execution_agent")
        builder.add_edge("execution_agent", "tool_call_beautifier")
        builder.add_conditional_edges("tool_call_beautifier", _route_after_llm)
        builder.add_edge("tools", "knowledge_agent")
        builder.add_edge("knowledge_agent", "execution_agent")

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
