from typing import Literal

from langgraph.graph import StateGraph, START, END

from backend.app.state import ETLState
from backend.app.agents import (
    request_understanding_agent,
    planning_agent,
    supervisor_agent,
    extraction_agent,
    profiling_agent,
    transformation_agent,
    validation_agent,
    loading_agent,
    diagnosis_agent,
    reporting_agent,
)


def ask_user_agent(state: ETLState) -> dict:

    question = (
        state.get("decision", {}).get("question")
        or "Please provide the missing information required to continue."
    )

    return {
        "status": "needs_human_input",
        "current_stage": "human_input",
        "final_report": question,
        "execution_log": [
            {
                "stage": "human_input",
                "message": question,
                "status": "waiting",
            }
        ],
    }


def route_supervisor(
    state: ETLState,
) -> Literal[
    "extract",
    "profile",
    "transform",
    "validate",
    "load",
    "diagnose",
    "report",
    "ask_user",
    "finish",
]:

    decision = state.get("decision", {})

    action = decision.get("next_action", "report")

    allowed_actions = {
        "extract",
        "profile",
        "transform",
        "validate",
        "load",
        "diagnose",
        "report",
        "ask_user",
        "finish",
    }

    if action not in allowed_actions:
        return "report"

    return action


def build_etl_graph():

    workflow = StateGraph(ETLState)

    # Core reasoning nodes
    workflow.add_node(
        "understanding",
        request_understanding_agent,
    )

    workflow.add_node(
        "planning",
        planning_agent,
    )

    workflow.add_node(
        "supervisor",
        supervisor_agent,
    )

    # Specialist agents
    workflow.add_node("extract", extraction_agent)

    workflow.add_node("profile", profiling_agent)

    workflow.add_node("transform", transformation_agent)

    workflow.add_node("validate", validation_agent)

    workflow.add_node("load", loading_agent)

    workflow.add_node("diagnose", diagnosis_agent)

    workflow.add_node("report", reporting_agent)

    workflow.add_node("ask_user", ask_user_agent)

    # Initial execution
    workflow.add_edge(START, "understanding")

    workflow.add_edge(
        "understanding",
        "planning",
    )

    workflow.add_edge(
        "planning",
        "supervisor",
    )

    # Supervisor dynamically selects the next agent
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "extract": "extract",
            "profile": "profile",
            "transform": "transform",
            "validate": "validate",
            "load": "load",
            "diagnose": "diagnose",
            "report": "report",
            "ask_user": "ask_user",
            "finish": END,
        },
    )

    # Every specialist returns control to supervisor
    specialist_agents = [
        "extract",
        "profile",
        "transform",
        "validate",
        "load",
        "diagnose",
    ]

    for agent in specialist_agents:
        workflow.add_edge(agent, "supervisor")

    # Reporting and clarification terminate this invocation
    workflow.add_edge("report", END)

    workflow.add_edge("ask_user", END)

    return workflow.compile()

# Compiled application
etl_graph = build_etl_graph()