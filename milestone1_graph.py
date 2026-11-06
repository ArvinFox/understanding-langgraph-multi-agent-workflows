from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class TeamState(TypedDict):
    task: str # The initial request from the user
    plan: str # Filled in by the Planner agent
    draft: str # Filled in by the Drafter agent

def planner_agent(state: TeamState) -> dict:
    print("Planner Agent Working...")
    user_task = state["task"] # Read 'task' from the whiteboard

    # Create a plan based on the task
    my_plan = f"1. Define both terms for '{user_task}'. 2. Compare pros/cons. 3. Give a real example."

    # Write ONLY 'plan' back to the whiteboard
    return {"plan": my_plan}

def drafter_agent(state: TeamState) -> dict:
    print("Drafter Agent Working...")
    current_plan = state["plan"] # Read the planner's output from the whiteboard

    # Create a draft following that plan
    my_draft = f"Here is the article written following the plan: [{current_plan}]"

    # Write ONLY 'draft' back to the  whiteboard
    return {"draft": my_draft}

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