import json
import uuid

from datetime import datetime, timezone
from typing import Any

#from langchain_google_genai import ChatGoogleGenerativeAI

#from backend.app.config import GOOGLE_API_KEY, GEMINI_MODEL
from backend.app.schemas import (
    RequestAnalysis,
    SupervisorDecision,
    TransformationPlan,
)
from backend.app.etl_tools import (
    extract_csv,
    profile_csv,
    execute_transformations,
    validate_csv,
    load_to_sqlite,
)


# llm = ChatGoogleGenerativeAI(
#     model=GEMINI_MODEL,
#     google_api_key=GOOGLE_API_KEY,
#     temperature=0,
# )

from langchain_groq import ChatGroq
from backend.app.config import GROQ_API_KEY, GROQ_MODEL

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
    max_retries=3,
)

def log_event(stage: str, message: str, status: str = "success") -> dict:

    return {
        "stage": stage,
        "message": message,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }



def request_understanding_agent(state: dict) -> dict:

    model = llm.with_structured_output(RequestAnalysis)

    result = model.invoke(
        f"""
        Understand the user's ETL request.

        User request:
        {state['user_request']}

        Source file:
        {state['source_path']}

        Destination table:
        {state['target_table']}

        Identify the main objective, requirements,
        and a high-level initial plan.

        Do not invent missing business requirements.
        """
    )

    return {
        "objective": result.objective,
        "requirements": result.requirements,
        "initial_plan": result.initial_plan,
        "run_id": str(uuid.uuid4()),
        "status": "running",
        "iteration": 0,
        "max_iterations": 20,
        "action_counts": {},
        "retry_count": 0,
        "execution_log": [
            log_event(
                "request_understanding",
                "User request understood.",
            )
        ],
    }


def planning_agent(state: dict) -> dict:

    return {
        "current_stage": "planning",
        "execution_log": [
            log_event(
                "planning",
                "Initial execution plan: "
                + json.dumps(state["initial_plan"]),
            )
        ],
    }

def supervisor_agent(state: dict) -> dict:

    iteration = state.get("iteration", 0) + 1

    if iteration > state.get("max_iterations", 20):

        return {
            "iteration": iteration,
            "status": "stopped",
            "decision": {
                "next_action": "report",
                "reasoning": "Maximum iteration limit reached.",
                "objective": state.get("objective", ""),
                "needs_human_input": False,
            },
            "execution_log": [
                log_event(
                    "supervisor",
                    "Maximum iteration limit reached.",
                    "warning",
                )
            ],
        }

    decision_model = llm.with_structured_output(
        SupervisorDecision
    )

    context = {
        "objective": state.get("objective"),
        "requirements": state.get("requirements"),
        "initial_plan": state.get("initial_plan"),
        "raw_artifact": state.get("raw_artifact"),
        "processed_artifact": state.get("processed_artifact"),
        "profile": state.get("profile"),
        "quality_issues": state.get("quality_issues"),
        "transformation_plan": state.get("transformation_plan"),
        "validation_result": state.get("validation_result"),
        "load_result": state.get("load_result"),
        "errors": state.get("errors", []),
        "action_counts": state.get("action_counts", {}),
        "status": state.get("status"),
        "diagnosis": state.get("diagnosis"),
    }

    result = decision_model.invoke(
        f"""
        You are the Supervisor Agent of an autonomous ETL system.

        Your responsibility is to decide which agent should
        execute NEXT based on the current state.

        Available agents:
        - extract: Extract source data.
        - profile: Analyze dataset structure and quality.
        - transform: Plan and execute required transformations.
        - validate: Validate the processed dataset.
        - load: Load validated data into SQLite.
        - diagnose: Diagnose errors and suggest recovery.
        - report: Generate the final execution report.
        - ask_user: Request missing business information.
        - finish: End execution when the objective is achieved.

        CURRENT STATE:
        {json.dumps(context, default=str, indent=2)}

        IMPORTANT RULES:

        1. Do not blindly follow the initial plan.
        2. Decide based on actual execution results.
        3. Do not profile before extraction.
        4. Do not transform before profiling.
        5. Do not load before successful validation.
        6. If validation fails, choose diagnosis or transformation.
        7. Skip unnecessary operations when evidence supports it.
        8. Ask the user if an essential business rule is unclear.
        9. Never repeatedly select the same failed action.
        10. If the task is complete, select report or finish.
        11. Never claim an operation succeeded unless state confirms it.
        12. Do not select load if validation has not passed.
        13. A failed load must not be blindly repeated.

        Select exactly one next action.
        Explain your reasoning.
        """
    )

    counts = dict(state.get("action_counts", {}))

    selected = result.next_action
    counts[selected] = counts.get(selected, 0) + 1

    # Safety limits for repeated actions.
    limits = {
        "extract": 1,
        "profile": 2,
        "transform": 3,
        "validate": 4,
        "load": 1,
        "diagnose": 3,
        "report": 1,
    }

    if counts[selected] > limits.get(selected, 20):
        selected = "report"
        reason = (
            f"Action limit reached for {result.next_action}. "
            "Stopping safely."
        )
    else:
        reason = result.reasoning

    # Enforce essential execution prerequisites independently
    # of the LLM's decision.
    if selected == "profile" and not state.get("raw_artifact"):
        selected = "extract"
        reason = "Cannot profile before extraction."

    if selected == "transform" and not state.get("profile"):
        selected = "profile"
        reason = "Cannot transform before profiling."

    if selected == "validate" and not state.get("processed_artifact"):
        selected = "transform"
        reason = "No processed artifact exists to validate."

    if selected == "load":
        validation = state.get("validation_result", {})
        if not validation.get("passed"):
            selected = "diagnose"
            reason = "Loading blocked because validation has not passed."

    if result.needs_human_input:
        selected = "ask_user"

    if state.get("load_result", {}).get("status") == "success":
        selected = "report"
        reason = "Loading succeeded. Generate the final report."

    decision = {
        "next_action": selected,
        "reasoning": reason,
        "objective": result.objective,
        "needs_human_input": result.needs_human_input,
        "question": result.question,
    }

    return {
        "iteration": iteration,
        "decision": decision,
        "action_counts": counts,
        "execution_log": [
            log_event(
                "supervisor",
                f"Selected {selected}: {reason}",
            )
        ],
    }




def extraction_agent(state: dict) -> dict:

    try:
        artifact = extract_csv(
            state["source_path"],
            state["run_id"],
        )

        return {
            "raw_artifact": artifact,
            "current_stage": "extraction",
            "execution_log": [
                log_event(
                    "extraction",
                    f"Extracted data to {artifact}",
                )
            ],
        }

    except Exception as exc:

        return {
            "errors": [str(exc)],
            "status": "error",
            "execution_log": [
                log_event(
                    "extraction",
                    str(exc),
                    "error",
                )
            ],
        }


def profiling_agent(state: dict) -> dict:

    try:
        result = profile_csv(state["raw_artifact"])

        return {
            "profile": result,
            "current_stage": "profiling",
            "execution_log": [
                log_event(
                    "profiling",
                    "Dataset profiling completed.",
                )
            ],
        }

    except Exception as exc:

        return {
            "errors": [str(exc)],
            "status": "error",
            "execution_log": [
                log_event("profiling", str(exc), "error")
            ],
        }


def transformation_agent(state: dict) -> dict:

    try:

        model = llm.with_structured_output(TransformationPlan)

        plan = model.invoke(
            f"""
            You are a Data Transformation Agent.

            Analyze this dataset profile:

            {json.dumps(state["profile"], indent=2)}

            User objective:
            {state["objective"]}

            Requirements:
            {json.dumps(state.get("requirements", []))}

            Choose only necessary transformations.

            Supported operations:
            - drop_duplicates
            - fill_numeric_missing
            - fill_text_missing
            - trim_whitespace
            - standardize_column_names

            Rules:
            - Do not invent columns.
            - Do not apply unnecessary operations.
            - Do not drop rows just because values are missing.
            - Use median for numeric missing values.
            - Use mode for text missing values.
            - If no transformation is needed, return an empty list.
            - Explain why each operation is necessary.
            """
        )

        operations = [
            operation.model_dump()
            for operation in plan.operations
        ]

        # No transformation needed: preserve the raw artifact.
        if not operations:

            return {
                "processed_artifact": state["raw_artifact"],
                "transformation_plan": [],
                "current_stage": "transformation",
                "execution_log": [
                    log_event(
                        "transformation",
                        "No transformations required.",
                    )
                ],
            }

        processed_file = execute_transformations(
            state["raw_artifact"],
            operations,
            state["run_id"],
        )

        return {
            "processed_artifact": processed_file,
            "transformation_plan": operations,
            "current_stage": "transformation",
            "execution_log": [
                log_event(
                    "transformation",
                    f"Executed {len(operations)} operations.",
                )
            ],
        }

    except Exception as exc:

        return {
            "errors": [str(exc)],
            "status": "error",
            "execution_log": [
                log_event(
                    "transformation",
                    str(exc),
                    "error",
                )
            ],
        }


def validation_agent(state: dict) -> dict:

    try:

        result = validate_csv(state["processed_artifact"])

        return {
            "validation_result": result,
            "current_stage": "validation",
            "status": "running" if result["passed"] else "validation_failed",
            "execution_log": [
                log_event(
                    "validation",
                    json.dumps(result),
                    "success" if result["passed"] else "warning",
                )
            ],
        }

    except Exception as exc:

        return {
            "errors": [str(exc)],
            "status": "error",
            "execution_log": [
                log_event("validation", str(exc), "error")
            ],
        }


def loading_agent(state: dict) -> dict:

    try:

        result = load_to_sqlite(
            state["processed_artifact"],
            state["target_table"],
        )

        return {
            "load_result": result,
            "current_stage": "loading",
            "status": "completed",
            "execution_log": [
                log_event(
                    "loading",
                    json.dumps(result),
                )
            ],
        }

    except Exception as exc:

        return {
            "errors": [str(exc)],
            "status": "load_failed",
            "execution_log": [
                log_event("loading", str(exc), "error")
            ],
        }

# error_diagnosis_agent
def diagnosis_agent(state: dict) -> dict:

    diagnosis = llm.invoke(
        f"""
        You are an ETL Error Diagnosis Agent.

        Objective:
        {state.get("objective")}

        Errors:
        {json.dumps(state.get("errors", []))}

        Validation results:
        {json.dumps(state.get("validation_result", {}))}

        Transformation plan:
        {json.dumps(state.get("transformation_plan", []))}

        Diagnose the root cause.

        Explain:
        1. What failed?
        2. Why did it fail?
        3. What should be changed?
        4. Is human intervention required?

        Do not claim that you fixed anything.
        """
    )

    return {
        "diagnosis": diagnosis.content,
        "current_stage": "diagnosis",
        "retry_count": state.get("retry_count", 0) + 1,
        "execution_log": [
            log_event(
                "diagnosis",
                str(diagnosis.content),
                "warning",
            )
        ],
    }


def reporting_agent(state: dict) -> dict:

    report = {
        "run_id": state.get("run_id"),
        "objective": state.get("objective"),
        "status": state.get("status"),
        "profile": state.get("profile"),
        "transformation_plan": state.get("transformation_plan"),
        "validation": state.get("validation_result"),
        "loading": state.get("load_result"),
        "errors": state.get("errors", []),
        "iterations": state.get("iteration"),
    }

    return {
        "final_report": json.dumps(
            report,
            indent=2,
            default=str,
        ),
        "current_stage": "reporting",
        "execution_log": [
            log_event("reporting", "Final report generated.")
        ],
    }