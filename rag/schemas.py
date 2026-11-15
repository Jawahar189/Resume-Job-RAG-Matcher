"""Pydantic models used both for LLM structured output and for the final report."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["technical", "tool", "soft_skill", "domain", "qualification"]
Importance = Literal["must_have", "nice_to_have"]
Status = Literal["matched", "partial", "missing"]
QuestionType = Literal["technical", "behavioral", "gap"]


def _normalize_literal(value: object) -> object:
    if isinstance(value, str):
        return value.strip().lower().replace(" ", "_").replace("-", "_")
    return value


# ---------- Step 1: job description -> requirements (LLM structured output) ----------


class Requirement(BaseModel):
    skill: str = Field(description="Short skill name, 1-4 words, e.g. 'Python' or 'REST API design'")
    requirement: str = Field(description="The requirement as phrased in the job description")
    category: Category = "technical"
    importance: Importance = "must_have"

    @field_validator("category", "importance", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        return _normalize_literal(value)


class JobRequirements(BaseModel):
    job_title: str
    requirements: list[Requirement]


# ---------- Step 3: grounded analysis (LLM structured output) ----------


class SkillAssessment(BaseModel):
    skill: str
    status: Status
    reason: str = Field(description="One sentence grounded in the resume evidence")

    @field_validator("status", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        return _normalize_literal(value)


class Improvement(BaseModel):
    section: str = Field(description="Resume section to change, e.g. 'Skills' or 'Experience'")
    suggestion: str
    example: str = Field(default="", description="Example bullet or rewrite the candidate could use")


class InterviewQuestion(BaseModel):
    question: str
    skill: str
    type: QuestionType
    purpose: str = Field(default="", description="What the interviewer wants to learn")

    @field_validator("type", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        return _normalize_literal(value)


class LLMAnalysis(BaseModel):
    summary: str
    skill_assessments: list[SkillAssessment]
    improvements: list[Improvement]
    interview_questions: list[InterviewQuestion]


# ---------- Final report shown in the UI ----------


class Evidence(BaseModel):
    text: str
    section: str
    score: float


class SkillMatch(BaseModel):
    skill: str
    requirement: str
    category: Category
    importance: Importance
    status: Status
    similarity: float
    keyword_hit: bool
    evidence: list[Evidence]
    reason: str = ""


class MatchReport(BaseModel):
    job_title: str
    overall_score: float
    coverage_score: float
    semantic_score: float
    skills: list[SkillMatch]
    summary: str
    improvements: list[Improvement]
    interview_questions: list[InterviewQuestion]
    used_llm: bool
    warnings: list[str] = []

    def by_status(self, status: Status) -> list[SkillMatch]:
        return [s for s in self.skills if s.status == status]
