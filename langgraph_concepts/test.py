from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    """This is an addition function that adds 2 integers together."""

    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtraction function for integers."""

    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplification function"""

    return a * b

tools = [add, subtract, multiply]

def model_call(state: AgentState) -> AgentState:
    system_promt = SystemMessage(
        content="You are my AI asistant, please answer my query to the best of your ability."
    )
    #response = llm.invoke([system_promt] + list(state["messages"]))
    print([system_promt])
    print(list(state["messages"]))
    print([system_promt] + list(state["messages"]))
    print([system_promt] + state["messages"])

    return {"messages": []}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
graph.add_edge(START, "our_agent")

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

inputs = {"messages": [("user", "Add 14 + 22")]}
model_call(inputs)