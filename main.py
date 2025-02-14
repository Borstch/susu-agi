from langgraph.types import Command

from susu_agi import SUSUAGI


def user_input() -> dict:
    question = input("> ")
    return {"messages": [("user", question)]}


def run_graph(graph, current_input, config):
    while True:
        for update in graph.stream(current_input, config=config, stream_mode="updates"):
            for node_id, value in update.items():
                if node_id == "__interrupt__":
                    print(value[0].value)
                    if type(value[0].value) == str:
                        return
                    continue

                if isinstance(value, dict) and value.get("messages", []):
                    last_message = value["messages"][-1]
                    if (
                        isinstance(last_message, dict)
                        or last_message.type != "ai"
                        or not last_message.content
                    ):
                        continue
                    print(f"{node_id}: {last_message.content}")

        if not graph.get_state(config).next:
            break

        feedback = input("FEEDBACK> ")
        if not feedback.strip():
            current_input = Command(resume={"action": "continue"})
        else:
            current_input = Command(
                resume={
                    "action": "feedback",
                    "data": f"User requested changes: {feedback}",
                }
            )


if __name__ == "__main__":
    agi = SUSUAGI(thread_id="1")

    with open("SUSU AGI.png", "wb") as dest:
        dest.write(agi._graph.get_graph().draw_mermaid_png())

    current_input = user_input()
    while True:
        output = agi.run_untill_interrupt(current_input)
        if (next_node := agi.get_next_node_name()) is None:
            break

        print(output.content or output.tool_calls)  # type: ignore
        match next_node:
            case "human_review":
                feedback = input("FEEDBACK> ")
                if not feedback.strip():
                    current_input = Command(resume={"action": "continue"})
                else:
                    current_input = Command(
                        resume={
                            "action": "feedback",
                            "data": f"User requested changes: {feedback}",
                        }
                    )
            case "next_question":
                current_input = Command(resume=input("> "))
            case _:
                raise ValueError(f"Got unknown node to run next: '{next_node}'.")
