from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenRouter(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    max_retries=1,
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

checkpoint = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpoint)

# initial_state = {
#     "messages": [HumanMessage(content="what is the capital of India during the British era")]
# }

# try:
#     response = chatbot.invoke(initial_state)
#     print(response["messages"][-1].content)
# except Exception as exc:
#     print(f"Chat model call failed: {exc}")
#     print("Try setting OPENROUTER_MODEL to another available model in your .env file.")

if __name__ == "__main__":
    thread_id = 1
    while True:
        user_message = input("Type here: ")
        print("User: ", user_message)
        if user_message.strip().lower() in ["exit", "quit", "bye"]:
            break
        config = {"configurable": {"thread_id": thread_id}}
        response = chatbot.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
        print("AI:", response["messages"][-1].content)
