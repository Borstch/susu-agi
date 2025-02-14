from abc import ABCMeta, abstractmethod

from langgraph.types import Command, interrupt
from typing_extensions import Literal

from state import AGIState


class HumanNodeBuilder(metaclass=ABCMeta):
    def __init__(self, approved_path: str, rejected_path: str):
        self._approved_path = approved_path
        self._rejected_path = rejected_path

        # HACK: We need to dynamiclly adjust types so GraphBuilder would know what to do
        def call_with_adjusted_types(
            state: AGIState,
        ) -> Command[Literal[approved_path, rejected_path]]:  # type: ignore
            return self(state)

        self.build = lambda: call_with_adjusted_types

    @abstractmethod
    def __call__(self, state: AGIState) -> Command:
        pass


class ReviewerBuilder(HumanNodeBuilder):
    def __call__(self, state: AGIState) -> Command:
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[-1]  # type: ignore

        human_review = interrupt(
            {
                "question": "Is this correct?",
                "tool_call": {"name": tool_call["name"], "args": tool_call["args"]},
            }
        )

        review_action = human_review["action"]
        if review_action == "continue":
            return Command(goto=self._approved_path)

        if review_action == "feedback":
            # NOTE: we're adding feedback message as a ToolMessage
            # to preserve the correct order in the message history
            # (AI messages with tool calls need to be followed by tool call messages)
            review_data = human_review["data"]
            tool_message = {
                "role": "tool",
                "content": review_data,
                "name": tool_call["name"],
                "tool_call_id": tool_call["id"],
            }
            return Command(
                goto=self._rejected_path, update={"messages": [tool_message]}
            )

        raise ValueError


class QuestionBuilder(HumanNodeBuilder):
    def __call__(self, state: AGIState) -> Command:
        question: str = interrupt(value="What is your next question?")
        if question.strip() in (":q", "quit", ""):
            return Command(goto=self._rejected_path)

        return Command(
            goto=self._approved_path,
            update={"messages": [{"role": "human", "content": question}]},
        )
