import operator

from typing import Annotated, Any, TypedDict


class ETLState(TypedDict, total=False):

    # User input
    user_request: str
    source_path: str
    target_table: str

    # Request understanding
    objective: str
    requirements: list[str]
    initial_plan: list[str]

    # Execution information
    run_id: str
    current_stage: str
    status: str

    # Artifact paths
    raw_artifact: str
    processed_artifact: str

    # Data intelligence
    profile: dict[str, Any]
    quality_issues: list[dict[str, Any]]
    transformation_plan: list[dict[str, Any]]

    # Validation
    validation_result: dict[str, Any]

    # Loading
    load_result: dict[str, Any]

    # Supervisor decision
    decision: dict[str, Any]
    action_counts: dict[str, int]

    # Error handling
    errors: list[str]
    retry_count: int

    # Execution limits
    iteration: int
    max_iterations: int

    # History
    execution_log: Annotated[list[dict[str, Any]], operator.add]

    # Final output
    final_report: str

    diagnosis: str