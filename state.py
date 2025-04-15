from langgraph.graph import MessagesState


class AGIState(MessagesState):
    knowledge: str
