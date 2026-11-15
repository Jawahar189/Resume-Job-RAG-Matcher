"""Command-line usage:

    python cli.py data/sample_resume.txt data/jd_ml_engineer.txt [more_jds...] [--json] [--no-llm]
"""

import argparse

from rag import ResumeMatcher
from rag.document_loader import load_file
from rag.llm import LLMClient

ICON = {"matched": "[x]", "partial": "[~]", "missing": "[ ]"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Match a resume against one or more job descriptions.")
    parser.add_argument("resume")
    parser.add_argument("job_descriptions", nargs="+")
    parser.add_argument("--json", action="store_true", help="print the structured report as JSON")
    parser.add_argument("--no-llm", action="store_true", help="ignore LLM settings in .env")
    args = parser.parse_args()

    matcher = ResumeMatcher(llm=None if args.no_llm else LLMClient.from_env())
    index = matcher.index_resume(load_file(args.resume))

    for path in args.job_descriptions:
        report = matcher.analyze(index, load_file(path))
        if args.json:
            print(report.model_dump_json(indent=2))
            continue
        print(f"\n=== {report.job_title} ({path}) ===")
        print(f"Overall {report.overall_score}% | coverage {report.coverage_score}% | "
              f"semantic {report.semantic_score}% | LLM used: {report.used_llm}")
        for w in report.warnings:
            print(f"! {w}")
        for s in report.skills:
            print(f"  {ICON[s.status]} {s.skill:<28} {s.importance:<13} sim={s.similarity:.2f} "
                  f"kw={'y' if s.keyword_hit else 'n'}")
        print(f"\n{report.summary}\n\nImprovements:")
        for imp in report.improvements:
            print(f"  - [{imp.section}] {imp.suggestion}")
        print("\nInterview questions:")
        for q in report.interview_questions:
            print(f"  - ({q.type}) {q.question}")


if __name__ == "__main__":
    main()
