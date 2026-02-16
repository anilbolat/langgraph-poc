from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    

system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

def drafter_node(state: AgentState) -> AgentState:
    """This node is responsible for drafting a document based on the provided messages."""

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_message = [system_prompt] + list(state["messages"]) + [user_message]

    response = llm.invoke(all_message)
    print(f"\n🤖 AI: {response.content}")

#    print(f"\n\n\nALL MSGS: {state['messages']}\n\n\n")

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Determines whether to continue drafting or to end the process."""
    messages = state["messages"]

    if not messages:
        return "continue"

    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and
            "document" in message.content.lower()):

            return "end"
        
    return "continue"


############# TOOL FUNCTIONS
@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content

    return f"Document has been updated successfully! The current content is:\n{document_content}"

@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """

    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
    
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully as '{filename}'!"
    
    except Exception as e:
        print(f"An error occurred while saving the document: {e}")
        return f"Failed to save the document: {e}"
    
tools = [update, save]

## Initialize the language model and bind the tools to it
llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite").bind_tools(tools)

# Graph construction
graph = StateGraph(AgentState)
graph.add_edge(START, "drafter")
graph.add_node("drafter", drafter_node)
graph.add_node("tools", ToolNode(tools=tools))
graph.add_edge("drafter", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "drafter",
        "end": END
    }
)

app = graph.compile()

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()