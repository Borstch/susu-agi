from langgraph.types import Command

from susu_agi import SUSUAGI


def user_input() -> dict:
    question = input("> ")
    return {"messages": [("user", question)]}


if __name__ == "__main__":
    agi = SUSUAGI(thread_id="1")

    with open("docs/SUSU_AGI.png", "wb") as dest:
        dest.write(agi._graph.get_graph().draw_mermaid_png())

    current_input = user_input()
    while True:
        output = agi.run_untill_interrupt(current_input)
        if (next_node := agi.get_next_node_name()) is None:
            break

        print(output.content)
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
