from typing import Literal

from pydantic import BaseModel, Field


class RequestAnalysis(BaseModel):
    objective: str
    requirements: list[str] = Field(default_factory=list)
    initial_plan: list[str] = Field(default_factory=list)


class SupervisorDecision(BaseModel):
    next_action: Literal[
        "extract",
        "profile",
        "transform",
        "validate",
        "load",
        "diagnose",
        "report",
        "ask_user",
        "finish",
    ]

    reasoning: str
    objective: str
    needs_human_input: bool = False
    question: str | None = None


class TransformOperation(BaseModel):
    operation: Literal[
        "drop_duplicates",
        "fill_numeric_missing",
        "fill_text_missing",
        "trim_whitespace",
        "standardize_column_names",
    ]

    columns: list[str] = Field(default_factory=list)
    reason: str


class TransformationPlan(BaseModel):
    operations: list[TransformOperation] = Field(default_factory=list)
    explanation: str