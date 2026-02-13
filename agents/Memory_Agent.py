from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv


load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite")

def process(state: AgentState) -> AgentState:
    """Process the agent's state and generate a response using the language model."""
    
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")

    state["messages"].append(AIMessage(content=response.content))
    print(f"Current State: {state['messages']}")

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

def load_conversation_history(filepath: str) -> List[Union[HumanMessage, AIMessage]]:
    """Load previous conversation history from a file."""
    messages = []
    try:
        with open(filepath, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("You: "):
                    messages.append(HumanMessage(content=line[5:]))
                elif line.startswith("AI: "):
                    messages.append(AIMessage(content=line[4:]))
    except FileNotFoundError:
        print("No previous conversation history found. Starting fresh.")
    return messages

conversation_history = load_conversation_history("conversation_history.txt")
if conversation_history:
    print(f"Loaded {len(conversation_history)} previous messages.")

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    response = agent.invoke({"messages": conversation_history})
    conversation_history = response["messages"]

    user_input = input("Enter: ")

with open("conversation_history.txt", "w") as file:
    file.write("Conversation History:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation\n")

print("Conversation history saved to conversation_history.txt")