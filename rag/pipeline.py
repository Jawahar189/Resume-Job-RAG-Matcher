"""End-to-end RAG pipeline.

    1. Index      resume -> chunks -> embeddings -> ChromaDB collection
    2. Extract    job description -> structured list of requirements (LLM or keyword fallback)
    3. Retrieve   for each requirement, hybrid search (vector + keyword) over resume chunks
    4. Score      similarity thresholds -> matched / partial / missing, plus an overall fit score
    5. Generate   LLM writes a grounded analysis from the retrieved evidence (or template fallback)
"""

from dataclasses import dataclass

import numpy as np

from . import config
from .chunking import Chunk, chunk_document
from .embeddings import Embedder, centroid, cosine
from .llm import LLMClient
from .schemas import (
    Evidence,
    Improvement,
    InterviewQuestion,
    JobRequirements,
    LLMAnalysis,
    MatchReport,
    Requirement,
    SkillMatch,
)
from .skills import extract_requirements_by_keywords, mentions
from .vector_store import Collection, SearchHit, VectorStore

EXTRACT_SYSTEM = (
    "You are an expert technical recruiter. Extract the concrete, checkable requirements from a job "
    "description: skills, tools, domain knowledge, soft skills and qualifications (degrees, years of "
    "experience). Use short canonical skill names (e.g. 'Python', 'Kubernetes', 'Stakeholder communication'). "
    "Mark items under 'preferred', 'nice to have', 'bonus' or 'a plus' as nice_to_have, everything else as "
    "must_have. Do not invent requirements and do not include company perks. Return at most 20 items."
)

ANALYSIS_SYSTEM = (
    "You are a senior technical recruiter and career coach. You compare a candidate's resume with a job "
    "description. You only see the parts of the resume retrieved as evidence for each requirement. "
    "Base every judgement strictly on that evidence; never assume experience that is not shown."
)

ANALYSIS_INSTRUCTIONS = """Tasks:
1. skill_assessments: one entry per requirement above, using the exact skill name. status is
   'matched' if the evidence clearly demonstrates it, 'partial' if only related or transferable experience
   is shown, 'missing' otherwise. The retrieval signals are hints, not the final answer.
2. summary: 2-3 sentences on overall fit, strongest points and biggest gaps.
3. improvements: 4-6 specific, actionable resume edits tied to this job. Prefer showing impact with numbers
   and surfacing relevant experience. For missing skills, advise adding them only if the candidate truly has
   them, or suggest a concrete way to build them (project, course). Include an example bullet where useful.
4. interview_questions: 8 questions an interviewer for this role would likely ask this candidate:
   about 4 'technical' questions probing matched skills in depth, 2-3 'gap' questions about missing or partial
   requirements, and 1-2 'behavioral' questions. Make them specific to the resume evidence."""


@dataclass
class ResumeIndex:
    text: str
    chunks: list[Chunk]
    embeddings: np.ndarray
    collection: Collection
    centroid: np.ndarray


class ResumeMatcher:
    def __init__(
        self,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
        top_k: int = config.TOP_K,
        match_threshold: float = config.MATCH_THRESHOLD,
        partial_threshold: float = config.PARTIAL_THRESHOLD,
    ):
        self.embedder = embedder or Embedder()
        self.llm = llm
        self.store = VectorStore()
        self.top_k = top_k
        self.match_threshold = match_threshold
        self.partial_threshold = partial_threshold

    # ---------- 1. Index ----------

    def index_resume(self, resume_text: str) -> ResumeIndex:
        chunks = chunk_document(resume_text, max_words=config.CHUNK_MAX_WORDS, prefix="resume")
        if not chunks:
            raise ValueError("The resume appears to be empty.")
        embeddings = self.embedder.embed([c.text for c in chunks])
        collection = self.store.create_collection(resume_text, chunks, embeddings)
        return ResumeIndex(resume_text, chunks, embeddings, collection, centroid(embeddings))

    # ---------- 2. Extract ----------

    def extract_requirements(self, jd_text: str, warnings: list[str]) -> JobRequirements:
        if self.llm:
            try:
                reqs = self.llm.structured(EXTRACT_SYSTEM, f"Job description:\n{jd_text[:8000]}", JobRequirements)
                if reqs.requirements:
                    return reqs
                warnings.append("LLM returned no requirements; used keyword extraction instead.")
            except Exception as exc:
                warnings.append(f"LLM requirement extraction failed ({exc}); used keyword extraction instead.")
        return extract_requirements_by_keywords(jd_text)

    # ---------- 3. Retrieve (hybrid) ----------

    def retrieve(self, index: ResumeIndex, req: Requirement) -> tuple[list[SearchHit], bool]:
        query_vec = self.embedder.embed_one(f"{req.skill}: {req.requirement}")
        hits = index.collection.search(query_vec, self.top_k)
        keyword_hit = mentions(req.skill, index.text)
        if keyword_hit:
            seen = {h.chunk_id for h in hits}
            lexical = [
                SearchHit(c.id, c.text, c.section, float(np.dot(index.embeddings[i], query_vec)))
                for i, c in enumerate(index.chunks)
                if c.id not in seen and mentions(req.skill, c.text)
            ]
            if lexical:
                hits.insert(0, max(lexical, key=lambda h: h.score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits, keyword_hit

    # ---------- 4. Score ----------

    def _status(self, similarity: float, keyword_hit: bool) -> str:
        if keyword_hit or similarity >= self.match_threshold:
            return "matched"
        if similarity >= self.partial_threshold:
            return "partial"
        return "missing"

    @staticmethod
    def coverage(skills: list[SkillMatch]) -> float:
        weight = {"must_have": 2.0, "nice_to_have": 1.0}
        credit = {"matched": 1.0, "partial": 0.5, "missing": 0.0}
        total = sum(weight[s.importance] for s in skills)
        if not total:
            return 0.0
        return sum(weight[s.importance] * credit[s.status] for s in skills) / total

    # ---------- Full analysis ----------

    def analyze(self, index: ResumeIndex, jd_text: str) -> MatchReport:
        warnings: list[str] = []
        job = self.extract_requirements(jd_text, warnings)
        if not job.requirements:
            warnings.append("No requirements could be extracted from this job description.")

        skills: list[SkillMatch] = []
        for req in job.requirements:
            hits, keyword_hit = self.retrieve(index, req)
            similarity = max((h.score for h in hits), default=0.0)
            skills.append(
                SkillMatch(
                    skill=req.skill,
                    requirement=req.requirement,
                    category=req.category,
                    importance=req.importance,
                    status=self._status(similarity, keyword_hit),
                    similarity=round(similarity, 3),
                    keyword_hit=keyword_hit,
                    evidence=[Evidence(text=h.text, section=h.section, score=round(h.score, 3)) for h in hits],
                )
            )

        jd_chunks = chunk_document(jd_text, max_words=config.CHUNK_MAX_WORDS, prefix="jd")
        jd_vec = centroid(self.embedder.embed([c.text for c in jd_chunks])) if jd_chunks else None
        semantic = max(0.0, cosine(index.centroid, jd_vec)) if jd_vec is not None else 0.0

        used_llm = False
        if self.llm and skills:
            try:
                analysis = self._llm_analysis(job.job_title, jd_text, skills)
                _apply_assessments(skills, analysis)
                summary, improvements, questions = analysis.summary, analysis.improvements, analysis.interview_questions
                used_llm = True
            except Exception as exc:
                warnings.append(f"LLM analysis failed ({exc}); showing rule-based suggestions.")
        if not used_llm:
            summary, improvements, questions = _template_analysis(job.job_title, skills, index.text)

        coverage = self.coverage(skills)
        overall = config.COVERAGE_WEIGHT * coverage + (1 - config.COVERAGE_WEIGHT) * semantic
        return MatchReport(
            job_title=job.job_title,
            overall_score=round(overall * 100, 1),
            coverage_score=round(coverage * 100, 1),
            semantic_score=round(semantic * 100, 1),
            skills=skills,
            summary=summary,
            improvements=improvements,
            interview_questions=questions,
            used_llm=used_llm,
            warnings=warnings,
        )

    # ---------- 5. Generate ----------

    def _llm_analysis(self, job_title: str, jd_text: str, skills: list[SkillMatch]) -> LLMAnalysis:
        lines = []
        for i, s in enumerate(skills, 1):
            lines.append(f'{i}. [{s.importance}] {s.skill} - "{s.requirement}"')
            lines.append(
                f"   retrieval: similarity={s.similarity:.2f}, keyword_found={'yes' if s.keyword_hit else 'no'}, "
                f"preliminary_status={s.status}"
            )
            for ev in s.evidence:
                lines.append(f'   evidence ({ev.section}, score {ev.score:.2f}): "{ev.text}"')
        user = (
            f"JOB TITLE: {job_title}\n\nJOB DESCRIPTION:\n{jd_text[:6000]}\n\n"
            f"REQUIREMENTS WITH RETRIEVED RESUME EVIDENCE:\n" + "\n".join(lines) + f"\n\n{ANALYSIS_INSTRUCTIONS}"
        )
        return self.llm.structured(ANALYSIS_SYSTEM, user, LLMAnalysis)


def _apply_assessments(skills: list[SkillMatch], analysis: LLMAnalysis) -> None:
    verdicts = {a.skill.strip().lower(): a for a in analysis.skill_assessments}
    for s in skills:
        verdict = verdicts.get(s.skill.strip().lower())
        if verdict:
            s.status = verdict.status
            s.reason = verdict.reason


def _template_analysis(
    job_title: str, skills: list[SkillMatch], resume_text: str
) -> tuple[str, list[Improvement], list[InterviewQuestion]]:
    matched = [s for s in skills if s.status == "matched"]
    partial = [s for s in skills if s.status == "partial"]
    missing = [s for s in skills if s.status == "missing"]
    missing_must = [s for s in missing if s.importance == "must_have"]

    def names(items: list[SkillMatch]) -> str:
        return ", ".join(s.skill for s in items[:5]) or "none"

    summary = (
        f"For '{job_title}', the resume shows {len(matched)} of {len(skills)} requirements clearly "
        f"({names(matched)}), {len(partial)} partially ({names(partial)}) and is missing {len(missing)} "
        f"({names(missing)})."
    )

    improvements: list[Improvement] = []
    for s in missing_must[:3]:
        improvements.append(
            Improvement(
                section="Skills / Experience",
                suggestion=f"'{s.skill}' is a must-have but isn't in your resume. If you have used it, add it to "
                f"your Skills section and show it in an Experience or Project bullet; if not, build a small "
                f"project with it.",
                example=f"Built <project> using {s.skill}, resulting in <measurable outcome>.",
            )
        )
    for s in partial[:2]:
        snippet = s.evidence[0].text[:120] if s.evidence else ""
        improvements.append(
            Improvement(
                section=s.evidence[0].section if s.evidence else "Experience",
                suggestion=f"Your experience related to '{s.skill}' is only implied. Name it explicitly so both "
                f"recruiters and ATS keyword filters pick it up.",
                example=f"Related text found: \"{snippet}...\"" if snippet else "",
            )
        )
    if sum(ch.isdigit() for ch in resume_text) < 10:
        improvements.append(
            Improvement(
                section="Experience",
                suggestion="Quantify your impact (%, time saved, users, revenue) - the resume has very few numbers.",
                example="Reduced API latency by 40% by introducing Redis caching for 2M daily requests.",
            )
        )

    questions: list[InterviewQuestion] = []
    for s in matched[:4]:
        questions.append(
            InterviewQuestion(
                question=f"Walk me through a project where you used {s.skill}. What was the hardest problem and "
                f"what trade-offs did you make?",
                skill=s.skill,
                type="technical",
                purpose=f"Verify depth of hands-on {s.skill} experience.",
            )
        )
    for s in (missing_must + partial)[:3]:
        questions.append(
            InterviewQuestion(
                question=f"This role needs {s.skill}. What is your experience with it or something similar, and "
                f"how would you get productive with it in your first month?",
                skill=s.skill,
                type="gap",
                purpose="Assess how the candidate handles a skill gap.",
            )
        )
    questions.append(
        InterviewQuestion(
            question="Tell me about a time you had to deliver under a tight deadline with unclear requirements.",
            skill="Problem Solving",
            type="behavioral",
            purpose="Assess prioritisation and communication.",
        )
    )
    return summary, improvements, questions
