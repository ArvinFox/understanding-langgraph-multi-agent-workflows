from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load GOOGLE_API_KEY from the .env file
load_dotenv()

# Initialize the LLM brain
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite",
    temperature=0.3,
    timeout=30,
    max_retries=2,)

class TeamState(TypedDict):
    task: str
    plan: str
    draft: str

def planner_agent(state: TeamState) -> dict:
    print("Planner Agent Thinking...")
    user_task = state["task"]

    messages = [
        SystemMessage(content="You are a Senior Technical Planner. Break the user's topic into a clear, numbered 3-step outline. Keep it brief and do NOT write the full article."),
        HumanMessage(content=f"Topic: {user_task}")
    ]
    response = llm.invoke(messages)

    # response.content holds the text string returned by the LLM
    return {"plan": response.text}

def drafter_agent(state: TeamState) -> dict:
    print("Drafter Agent Thinking...")
    user_task = state["task"]
    plan = state["plan"]
    messages = [
        SystemMessage(content="You are a Technical Writer. Write a clear, concise explanation strictly following the provided plan."),
        HumanMessage(content=f"Task: {user_task}\n\nPlan:\n{plan}")
    ]
    
    response = llm.invoke(messages)
    return {"draft": response.text}

# 1. Initialize the graph with our whiteboard schema
builder = StateGraph(TeamState)

# 2. Register each agent function and give it a node name
builder.add_node("planner", planner_agent)
builder.add_node("drafter", drafter_agent)

# 3. Connect the workflow arrows (Edges)
builder.add_edge(START, "planner") # When graph starts -> go to "planner"
builder.add_edge("planner", "drafter") # After "planner" finishes -> go to "drafter"
builder.add_edge("drafter", END) # After "drafter" finishes -> stop (END)

# 4. Lock in (compile) the graph so it's ready to run
graph = builder.compile()

# Start the workflow by filling in the initial 'task' slot
final_state = graph.invoke({"task": "Expain microservices vs monoliths"})

print("\n=== FINAL WHITEBOARD STATE ===")
print("Task :", final_state["task"])
print("Plan :", final_state["plan"])
print("Draft :", final_state["draft"])
    