"""Skill taxonomy + keyword matching.

Used for two things:
1. Lexical half of hybrid retrieval: if a skill (or alias) literally appears in
   the resume, that's strong evidence regardless of embedding similarity.
2. Fallback requirement extraction when no LLM is configured.
"""

import re

from .schemas import Category, JobRequirements, Requirement

# canonical name -> (category, aliases). Aliases are matched case-insensitively.
SKILLS: dict[str, tuple[Category, list[str]]] = {
    # Programming languages
    "Python": ("technical", ["python"]),
    "Java": ("technical", ["java"]),
    "JavaScript": ("technical", ["javascript", "js", "ecmascript"]),
    "TypeScript": ("technical", ["typescript"]),
    "C++": ("technical", ["c++", "cpp"]),
    "C#": ("technical", ["c#", "csharp"]),
    "Go": ("technical", ["golang"]),
    "Rust": ("technical", ["rust"]),
    "Kotlin": ("technical", ["kotlin"]),
    "Swift": ("technical", ["swift"]),
    "PHP": ("technical", ["php"]),
    "Ruby": ("technical", ["ruby", "ruby on rails", "rails"]),
    "Scala": ("technical", ["scala"]),
    "SQL": ("technical", ["sql"]),
    "Bash": ("technical", ["bash", "shell scripting"]),
    # Web
    "HTML": ("technical", ["html", "html5"]),
    "CSS": ("technical", ["css", "css3", "sass", "scss"]),
    "React": ("tool", ["react", "react.js", "reactjs"]),
    "Next.js": ("tool", ["next.js", "nextjs"]),
    "Angular": ("tool", ["angular"]),
    "Vue.js": ("tool", ["vue", "vue.js", "vuejs"]),
    "Redux": ("tool", ["redux"]),
    "Tailwind CSS": ("tool", ["tailwind", "tailwind css"]),
    "Node.js": ("tool", ["node.js", "nodejs", "node"]),
    "Express": ("tool", ["express.js", "expressjs"]),
    "Django": ("tool", ["django"]),
    "Flask": ("tool", ["flask"]),
    "FastAPI": ("tool", ["fastapi"]),
    "Spring Boot": ("tool", ["spring boot"]),
    "REST APIs": ("technical", ["restful", "rest api", "rest apis"]),
    "GraphQL": ("technical", ["graphql"]),
    "Web Accessibility": ("technical", ["accessibility", "a11y", "wcag"]),
    "Jest": ("tool", ["jest"]),
    "Webpack": ("tool", ["webpack", "vite"]),
    # Data / ML / AI
    "Machine Learning": ("technical", ["machine learning", "ml"]),
    "Deep Learning": ("technical", ["deep learning", "neural networks"]),
    "NLP": ("technical", ["nlp", "natural language processing"]),
    "Computer Vision": ("technical", ["computer vision", "opencv"]),
    "LLMs": ("technical", ["llm", "llms", "large language models", "gpt"]),
    "RAG": ("technical", ["rag", "retrieval-augmented generation", "retrieval augmented generation"]),
    "Prompt Engineering": ("technical", ["prompt engineering"]),
    "LangChain": ("tool", ["langchain"]),
    "LlamaIndex": ("tool", ["llamaindex", "llama index"]),
    "Hugging Face": ("tool", ["hugging face", "huggingface", "transformers"]),
    "Vector Databases": ("tool", ["vector database", "vector databases", "pinecone", "chromadb", "chroma", "faiss", "weaviate", "qdrant", "pgvector"]),
    "PyTorch": ("tool", ["pytorch", "torch"]),
    "TensorFlow": ("tool", ["tensorflow", "keras"]),
    "scikit-learn": ("tool", ["scikit-learn", "sklearn"]),
    "Pandas": ("tool", ["pandas"]),
    "NumPy": ("tool", ["numpy"]),
    "Spark": ("tool", ["spark", "pyspark", "apache spark"]),
    "Airflow": ("tool", ["airflow"]),
    "MLOps": ("technical", ["mlops", "mlflow", "model deployment", "model monitoring"]),
    "Statistics": ("technical", ["statistics", "statistical", "a/b testing", "hypothesis testing"]),
    "Data Visualization": ("technical", ["data visualization", "tableau", "power bi", "matplotlib", "seaborn"]),
    "ETL": ("technical", ["etl", "elt", "data pipelines", "data pipeline"]),
    # Databases
    "PostgreSQL": ("tool", ["postgresql", "postgres"]),
    "MySQL": ("tool", ["mysql"]),
    "MongoDB": ("tool", ["mongodb", "mongo"]),
    "Redis": ("tool", ["redis"]),
    "Elasticsearch": ("tool", ["elasticsearch", "opensearch"]),
    "Snowflake": ("tool", ["snowflake"]),
    # Cloud / DevOps
    "AWS": ("tool", ["aws", "amazon web services", "ec2", "s3", "lambda", "sagemaker"]),
    "Azure": ("tool", ["azure"]),
    "GCP": ("tool", ["gcp", "google cloud", "bigquery", "vertex ai"]),
    "Docker": ("tool", ["docker", "containers", "containerization"]),
    "Kubernetes": ("tool", ["kubernetes", "k8s", "eks", "gke", "aks"]),
    "Terraform": ("tool", ["terraform", "infrastructure as code"]),
    "CI/CD": ("technical", ["ci/cd", "continuous integration", "github actions", "jenkins", "gitlab ci"]),
    "Git": ("tool", ["git", "github", "gitlab"]),
    "Linux": ("technical", ["linux", "unix"]),
    "Microservices": ("technical", ["microservices", "microservice"]),
    "System Design": ("technical", ["system design", "distributed systems", "scalable systems"]),
    "Testing": ("technical", ["unit testing", "pytest", "test automation", "tdd", "integration testing"]),
    # Soft skills / ways of working
    "Communication": ("soft_skill", ["communication", "communicate", "stakeholder"]),
    "Leadership": ("soft_skill", ["leadership", "led a team", "mentoring", "mentored", "mentor"]),
    "Teamwork": ("soft_skill", ["teamwork", "collaboration", "collaborated", "cross-functional"]),
    "Problem Solving": ("soft_skill", ["problem solving", "problem-solving", "analytical"]),
    "Agile": ("soft_skill", ["agile", "scrum", "kanban"]),
}

_NICE_TO_HAVE = re.compile(r"nice to have|preferred|bonus|a plus|is a plus|good to have|desirable|optional", re.I)
_MUST_HAVE = re.compile(r"requirements|required|must have|qualifications|what you bring|you have", re.I)


def _alias_pattern(alias: str) -> re.Pattern:
    return re.compile(r"(?<![\w+#.])" + re.escape(alias) + r"(?![\w+#])", re.I)


_PATTERNS: dict[str, list[re.Pattern]] = {
    name: [_alias_pattern(a) for a in aliases] for name, (_, aliases) in SKILLS.items()
}
_LOOKUP = {name.lower(): name for name in SKILLS}
for _name, (_, _aliases) in SKILLS.items():
    for _alias in _aliases:
        _LOOKUP.setdefault(_alias.lower(), _name)


def patterns_for(skill: str) -> list[re.Pattern]:
    """Patterns for a skill name, using taxonomy aliases when the skill is known."""
    canonical = _LOOKUP.get(skill.strip().lower())
    if canonical:
        return _PATTERNS[canonical]
    return [_alias_pattern(skill.strip())]


def mentions(skill: str, text: str) -> bool:
    return any(p.search(text) for p in patterns_for(skill))


def extract_requirements_by_keywords(jd_text: str) -> JobRequirements:
    """No-LLM fallback: scan the JD line by line for known skills."""
    requirements: dict[str, Requirement] = {}
    section_is_optional = False
    for line in jd_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_heading = len(stripped.split()) <= 6 and not stripped.endswith(".")
        if is_heading and _NICE_TO_HAVE.search(stripped):
            section_is_optional = True
        elif is_heading and _MUST_HAVE.search(stripped):
            section_is_optional = False
        optional = section_is_optional or bool(_NICE_TO_HAVE.search(stripped))
        for name, (category, _) in SKILLS.items():
            if name in requirements or not any(p.search(stripped) for p in _PATTERNS[name]):
                continue
            requirements[name] = Requirement(
                skill=name,
                requirement=stripped.lstrip("-*\u2022 ").strip(),
                category=category,
                importance="nice_to_have" if optional else "must_have",
            )
    title = next((ln.strip() for ln in jd_text.splitlines() if ln.strip()), "Job description")
    return JobRequirements(job_title=title[:80], requirements=list(requirements.values()))
