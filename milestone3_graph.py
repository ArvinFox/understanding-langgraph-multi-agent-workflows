from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

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
    research_notes: str # NEW: Filled in by the Researcher Agent
    draft: str

@tool
def lookup_company_standards(topic: str) -> str:
    """Looks up internal AcmeCorp engineering standards and metrics for a given technical topic."""
    print(f" [TOOL EXECUTED] Searching internal database for: '{topic}'...")
    return(
        "AcmeCorp Official Standards (2026):\n"
        "- Monoliths must use modular Python packages and PostgreSQL 16.\n"
        "- A team is ONLY allowed to split into Microservices if traffic exceeds 10,000 requests/sec "
        "or the engineering team has more than 25 developers.\n"
        "- All microservices must communicate via gRPC and deploy on GKE."
    )

# Create a version of the LLM that knows about this tool
llm_with_tools = llm.bind_tools([lookup_company_standards])

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

def researcher_agent(state: TeamState) -> dict:
    print("Researcher Agent Thinking...")
    user_task = state["task"]
    message = [
        SystemMessage(content="You are a Technical Researcher. Always use the lookup_company_standards tool to fetch internal company rules for the user's topic."),
        HumanMessage(content=f"Find internal standards for: {user_task}")
    ]

    # 1. Ask the tool-enabled LLM what to do
    ai_msg = llm_with_tools.invoke(message)

    # 2. Check if the LLM requested a tool call
    if ai_msg.tool_calls:
        tool_call = ai_msg.tool_calls[0] # e.g., {'name': 'lookup_company_standards', 'args': {'topic': '...'}}
        tool_output = lookup_company_standards.invoke(tool_call["args"])
        return {"research_notes": tool_output}

    # Fallback if no tool was called
    return {"research_notes": ai_msg.text}

def drafter_agent(state: TeamState) -> dict:
    print("Drafter Agent Thinking...")
    user_task = state["task"]
    plan = state["plan"]
    research_notes = state["research_notes"]
    messages = [
        SystemMessage(content="You are a Technical Writer. Write a clear, concise explanation strictly following the provided plan."),
        HumanMessage(content=f"Task: {user_task}\n\nPlan:\n{plan}\n\nInternal Company Standards: \n{research_notes}")
    ]
    
    response = llm.invoke(messages)
    return {"draft": response.text}

# 1. Initialize the graph with our whiteboard schema
builder = StateGraph(TeamState)

# 2. Register each agent function and give it a node name
builder.add_node("planner", planner_agent)
builder.add_node("researcher", researcher_agent)
builder.add_node("drafter", drafter_agent)

# 3. Connect the workflow arrows (Edges)
builder.add_edge(START, "planner") # When graph starts -> go to "planner"
builder.add_edge("planner", "researcher") # After "planner" finishes -> go to "researcher"
builder.add_edge("researcher", "drafter") # After "researcher" finishes -> go to "drafter"
builder.add_edge("drafter", END) # After "drafter" finishes -> stop (END)

# 4. Lock in (compile) the graph so it's ready to run
graph = builder.compile()

# Start the workflow by filling in the initial 'task' slot
final_state = graph.invoke({"task": "Expain microservices vs monoliths"})

print("\n=== FINAL WHITEBOARD STATE ===")
print("Task :", final_state["task"])
print("Plan :", final_state["plan"])
print("Research :", final_state["research_notes"])
print("Draft :", final_state["draft"])
    