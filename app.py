"""Streamlit UI.  Run with:  streamlit run app.py"""

from pathlib import Path

import streamlit as st

from rag import ResumeMatcher
from rag import config
from rag.document_loader import SUPPORTED_EXTENSIONS, clean_text, load_text
from rag.embeddings import Embedder
from rag.llm import LLMClient
from rag.schemas import MatchReport, SkillMatch

DATA_DIR = Path(__file__).parent / "data"
JD_SEPARATOR = "---"
STATUS_ICON = {"matched": "✅", "partial": "🟡", "missing": "❌"}

st.set_page_config(page_title="Resume ↔ Job Matcher (RAG)", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~80 MB)...")
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource
def get_matcher(use_llm: bool, top_k: int, match_t: float, partial_t: float) -> ResumeMatcher:
    return ResumeMatcher(
        embedder=get_embedder(),
        llm=LLMClient.from_env() if use_llm else None,
        top_k=top_k,
        match_threshold=match_t,
        partial_threshold=partial_t,
    )


def load_sample() -> None:
    st.session_state.resume_input = (DATA_DIR / "sample_resume.txt").read_text(encoding="utf-8")
    st.session_state.jd_input = f"\n{JD_SEPARATOR}\n".join(
        (DATA_DIR / name).read_text(encoding="utf-8") for name in ("jd_ml_engineer.txt", "jd_frontend_developer.txt")
    )
    st.session_state.pop("reports", None)


# ---------------- Sidebar ----------------

with st.sidebar:
    st.header("Settings")
    llm_available = LLMClient.from_env() is not None
    if llm_available:
        use_llm = st.toggle(f"Use LLM ({config.LLM_MODEL})", value=True)
    else:
        use_llm = False
        st.info("No LLM configured: running in rule-based mode (keywords + embeddings + templates). "
                "Copy `.env.example` to `.env` to enable LLM analysis.")
    top_k = st.slider("Chunks retrieved per requirement (top-k)", 1, 6, config.TOP_K)
    match_t = st.slider("Match threshold (cosine)", 0.30, 0.90, config.MATCH_THRESHOLD, 0.01)
    partial_t = st.slider("Partial threshold (cosine)", 0.10, 0.80, config.PARTIAL_THRESHOLD, 0.01)
    st.button("Load sample resume + 2 jobs", on_click=load_sample, use_container_width=True)
    st.caption(f"Embedding model: `{config.EMBEDDING_MODEL}`")

# ---------------- Inputs ----------------

st.title("📄 Resume ↔ Job Description Matcher")
st.caption("RAG pipeline: chunk & embed resume → store in ChromaDB → extract job requirements → "
           "retrieve resume evidence per requirement → score → LLM generates grounded, structured analysis.")

left, right = st.columns(2)
with left:
    st.subheader("1. Resume")
    resume_file = st.file_uploader("Upload resume", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS])
    st.text_area("...or paste resume text", key="resume_input", height=260)
with right:
    st.subheader("2. Job description(s)")
    jd_files = st.file_uploader("Upload one or more job descriptions",
                                type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS], accept_multiple_files=True)
    st.text_area(f"...or paste job descriptions (separate several with a line containing only `{JD_SEPARATOR}`)",
                 key="jd_input", height=260)


def collect_inputs() -> tuple[str, list[str]]:
    resume = load_text(resume_file.getvalue(), resume_file.name) if resume_file else clean_text(
        st.session_state.get("resume_input", ""))
    jds = [load_text(f.getvalue(), f.name) for f in jd_files or []]
    pasted = st.session_state.get("jd_input", "")
    jds += [clean_text(p) for p in pasted.split(f"\n{JD_SEPARATOR}\n") if p.strip()]
    return resume, jds


if st.button("🔍 Analyze", type="primary", use_container_width=True):
    try:
        resume_text, jd_texts = collect_inputs()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if not resume_text or not jd_texts:
        st.warning("Please provide a resume and at least one job description.")
        st.stop()
    matcher = get_matcher(use_llm, top_k, match_t, partial_t)
    with st.spinner("Indexing resume..."):
        index = matcher.index_resume(resume_text)
    reports: list[MatchReport] = []
    progress = st.progress(0.0, text="Analyzing job descriptions...")
    for i, jd in enumerate(jd_texts, 1):
        reports.append(matcher.analyze(index, jd))
        progress.progress(i / len(jd_texts), text=f"Analyzed {i}/{len(jd_texts)}")
    progress.empty()
    st.session_state.reports = reports
    st.session_state.chunks = [{"id": c.id, "section": c.section, "text": c.text} for c in index.chunks]

# ---------------- Results ----------------


def render_skill(s: SkillMatch) -> None:
    label = f"{STATUS_ICON[s.status]} {s.skill}" + ("" if s.importance == "must_have" else "  · nice to have")
    with st.expander(label):
        st.markdown(f"**Requirement:** {s.requirement}")
        st.markdown(f"**Best similarity:** `{s.similarity:.2f}` · **Keyword found:** {'yes' if s.keyword_hit else 'no'}")
        if s.reason:
            st.markdown(f"**Why:** {s.reason}")
        st.markdown("**Retrieved resume evidence:**")
        for ev in s.evidence:
            st.markdown(f"- `{ev.score:.2f}` *{ev.section}* — {ev.text}")


def render_report(report: MatchReport, position: int) -> None:
    for w in report.warnings:
        st.warning(w)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall fit", f"{report.overall_score:.0f}%")
    c2.metric("Requirement coverage", f"{report.coverage_score:.0f}%")
    c3.metric("Semantic similarity", f"{report.semantic_score:.0f}%")
    c4.metric("Matched / total", f"{len(report.by_status('matched'))} / {len(report.skills)}")
    st.progress(min(report.overall_score / 100, 1.0))
    st.markdown(f"**Summary.** {report.summary}")
    st.caption("Analysis generated by LLM from retrieved evidence." if report.used_llm
               else "Rule-based analysis (no LLM).")

    cols = st.columns(3)
    for col, status, title in zip(cols, ("matched", "partial", "missing"),
                                  ("Matching skills", "Partial matches", "Missing skills")):
        with col:
            items = report.by_status(status)
            st.subheader(f"{STATUS_ICON[status]} {title} ({len(items)})")
            for s in sorted(items, key=lambda x: x.importance):
                render_skill(s)

    st.subheader("💡 Suggested improvements")
    for imp in report.improvements:
        st.markdown(f"- **[{imp.section}]** {imp.suggestion}")
        if imp.example:
            st.markdown(f"    > {imp.example}")

    st.subheader("🎤 Likely interview questions")
    for qtype in ("technical", "gap", "behavioral"):
        qs = [q for q in report.interview_questions if q.type == qtype]
        if qs:
            st.markdown(f"**{qtype.title()}**")
            for q in qs:
                st.markdown(f"- {q.question}  \n  <small>skill: {q.skill}. {q.purpose}</small>", unsafe_allow_html=True)

    st.download_button("⬇️ Download structured report (JSON)", report.model_dump_json(indent=2),
                       file_name=f"report_{report.job_title[:30].replace(' ', '_')}.json", mime="application/json",
                       key=f"download-{position}")


reports: list[MatchReport] = st.session_state.get("reports", [])
if reports:
    st.divider()
    st.header("Results")
    if len(reports) > 1:
        st.subheader("Ranking")
        ranked = sorted(reports, key=lambda r: r.overall_score, reverse=True)
        st.dataframe(
            [{"Job": r.job_title, "Overall %": r.overall_score, "Coverage %": r.coverage_score,
              "Semantic %": r.semantic_score, "Matched": len(r.by_status("matched")),
              "Missing": len(r.by_status("missing"))} for r in ranked],
            use_container_width=True, hide_index=True,
        )
    tabs = st.tabs([f"{i + 1}. {r.job_title[:40]}" for i, r in enumerate(reports)])
    for position, (tab, report) in enumerate(zip(tabs, reports)):
        with tab:
            render_report(report, position)

    with st.expander("🔎 RAG internals: resume chunks stored in the vector database"):
        st.dataframe(st.session_state.get("chunks", []), use_container_width=True, hide_index=True)
