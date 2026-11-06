from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver

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
    research_notes: str
    draft: str
    feedback: str # NEW: Critique from the Reviewer
    approved: bool # NEW: True if Reviewer approves, False if Reviewer rejects
    revision_count: int # NEW: Safety counter so we never loop forever

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

class ReviewDecision(BaseModel):
    approved: bool = Field(description="True ONLY if the draft meets all rules, False otherwise.")
    feedback: str = Field(description="Specific instructions on what to fix if rejected, or praise if approved.")

# This special LLM version is guaranteed to return a ReviewDecision object!
reviewer_llm = llm.with_structured_output(ReviewDecision)

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
    # Get current revision count (defaults to 0 on first run) and add 1
    revisions = state.get("revision_count", 0) + 1
    feedback = state.get("feedback", "")
    print(f"Drafter Agent Thinking (Revision # {revisions})...")

    user_task = state["task"]
    plan = state["plan"]
    research_notes = state["research_notes"]

    prompt = f"Task: {user_task}\n\nPlan:\n{plan}\n\nInternal Company Standards:\n{research_notes}"

    if feedback:
        previous_draft = state.get("draft", "")
        prompt += f"\n\nYOUR PREVIOUS DRAFT:\n{previous_draft}\n\nIMPORTANT - FIX THIS REVIEWER FEEDBACK:\n{feedback}"
    
    messages = [
        SystemMessage(content="You are a Technical Writer. Write a clear, concise explanation strictly following the plan, company standards, and any reviewer feedback."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    return {"draft": response.text, "revision_count": revisions}

def reviewer_agent(state: TeamState) -> dict:
    print("Reviewer Agent Inspecting Draft...")
    draft = state["draft"]

    messages = [
        SystemMessage(content=(
            "You are a Strict Principal Engineer Reviewer. "
            "Check the draft against TWO mandatory rules:\n"
            "1. It must mention AcmeCorp's 10,000 requests/sec and PostgreSQL 16 standards.\n"
            "2. It MUST include a short ```python code block (written in Python, NOT protobuf) showing a sample service call.\n"
            "If EITHER rule is missing, set approved=False and tell the writer exactly what to add!"
        )),
        HumanMessage(content=f"Here is the draft to review:\n\n{draft}")
    ]

    # reviewer_llm returns a ReviewDecision object (.approved and .feedback)
    decision = reviewer_llm.invoke(messages)
    print(f"[REVIEW RESULT] Approved: {decision.approved} | Feedback: {decision.feedback}")

    return {"approved": decision.approved, "feedback": decision.feedback}

def route_after_review(state: TeamState) -> str:
    """Traffic cop: decides whether to finish or loop back to the drafter."""
    if state["approved"]:
        print("[ROUTER] Draft approved! Finishing workflow.")
        return END
    if state["revision_count"] >= 2:
        print("[ROUTER] Max revisions (2) reached! Stopping to prevent infinite loop.")
        return END
    print("   [ROUTER] Draft rejected! Sending back to Drafter...")
    return "drafter"

# 1. Initialize the graph with our whiteboard schema
builder = StateGraph(TeamState)

# 2. Register each agent function and give it a node name
builder.add_node("planner", planner_agent)
builder.add_node("researcher", researcher_agent)
builder.add_node("drafter", drafter_agent)
builder.add_node("reviewer", reviewer_agent)

# 3. Connect the workflow arrows (Edges)
builder.add_edge(START, "planner") # When graph starts -> go to "planner"
builder.add_edge("planner", "researcher") # After "planner" finishes -> go to "researcher"
builder.add_edge("researcher", "drafter") # After "researcher" finishes -> go to "drafter"
builder.add_edge("drafter", "reviewer") # After "drafter" finishes -> go to "reviewer"
builder.add_conditional_edges("reviewer", route_after_review)

# 4. Attach a Checkpointer so Langgraph saves state after every agent step
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 5. Create a session thread_id (in production, this is a user's session or ticket ID)
config = {"configurable": {"thread_id": "acmecorp-session-001"}}

user_topic = input("\nEnter a technical topic for the Agent Team (or press Enter for default): ").strip()
if not user_topic:
    user_topic = "Explain microservices vs monoliths"
print(f"\nStarting Multi-Agent Workflow for: '{user_topic}'\n" + "=" * 60)

# Stream each agent's whiteboard update in real time as soon as that agent finishes!
for event in graph.stream({"task": user_topic}, config=config, stream_mode="updates"):
    for agent_name, state_update in event.items():
        print(f"\n[{agent_name.upper()} FINISHED] Updated keys: {list(state_update.keys())}")

# 6. Fetch the persisted final state directly from the Checkpointer memory!
saved_snapshot = graph.get_state(config)
final_state = saved_snapshot.values

print("\n" + "=" * 60)
print("FINAL STATE RETRIEVED FROM CHECKPOINTER MEMORY (Thread: acmecorp-session-001)")
print("=" * 60)
print("Task            :", final_state["task"])
print("Revision Count  :", final_state["revision_count"])
print("Status (Approved):", final_state["approved"])
print("\n--- FINAL APPROVED DRAFT ---\n")
print(final_state["draft"])
