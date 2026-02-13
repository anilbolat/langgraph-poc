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

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_promt = SystemMessage(
        content="You are my AI asistant, please answer my query to the best of your ability."
    )
    response = llm.invoke([system_promt] + list(state["messages"]))

    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

###
# START ----> our_agent ----> should_continue ----> (tools or END)
# ... tools ----> our_agent -> should_continue -> (tools or END) -> ...
# 
# there is a loop between our_agent and tools, where the agent can call a tool, get the result, 
# and then decide whether to call another tool or end the conversation based on the presence of tool calls 
# in the last message.
#
###


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

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 14 + 22 and then multiply the result by 3.")]}
print_stream(app.stream(inputs, stream_mode="values"))