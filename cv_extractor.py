"""
cv_extractor.py
===============
Rule-based CV candidate extraction module.

Converts raw CV text into structured evidence candidates using deterministic,
regex-and-heuristic-based methods. No LLMs, no ML models, no hallucination.

Usage:
    from cv_extractor import extract_candidates
    result = extract_candidates(raw_text)
    # result["candidates"] -> list of candidate dicts
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Section header aliases → normalized internal section names
SECTION_ALIASES: dict[str, str] = {
    # contact / personal
    "contact": "contact",
    "contact information": "contact",
    "personal information": "contact",
    "personal details": "contact",
    "personal data": "contact",
    # summary / objective
    "summary": "summary",
    "professional summary": "summary",
    "career summary": "summary",
    "profile": "summary",
    "professional profile": "summary",
    "objective": "summary",
    "career objective": "summary",
    "about me": "summary",
    "about": "summary",
    "introduction": "summary",
    "bio": "summary",
    # skills
    "skills": "skills",
    "technical skills": "skills",
    "core skills": "skills",
    "key skills": "skills",
    "competencies": "skills",
    "core competencies": "skills",
    "technologies": "skills",
    "tech stack": "skills",
    "tools": "skills",
    "tools & technologies": "skills",
    "technical competencies": "skills",
    "programming skills": "skills",
    "software skills": "skills",
    "professional skills": "skills",
    "skill set": "skills",
    "areas of expertise": "skills",
    # experience
    "experience": "experience",
    "work experience": "experience",
    "professional experience": "experience",
    "employment history": "experience",
    "employment": "experience",
    "career history": "experience",
    "work history": "experience",
    "relevant experience": "experience",
    "internship": "experience",
    "internships": "experience",
    "volunteer experience": "experience",
    "freelance experience": "experience",
    # education
    "education": "education",
    "educational background": "education",
    "academic background": "education",
    "academic qualifications": "education",
    "qualifications": "education",
    "academic history": "education",
    # projects
    "projects": "projects",
    "personal projects": "projects",
    "academic projects": "projects",
    "selected projects": "projects",
    "key projects": "projects",
    "portfolio": "projects",
    # certifications
    "certifications": "certifications",
    "certification": "certifications",
    "licenses": "certifications",
    "licenses & certifications": "certifications",
    "professional certifications": "certifications",
    "accreditations": "certifications",
    # training
    "training": "training",
    "courses": "training",
    "training & courses": "training",
    "courses & training": "training",
    "training and courses": "training",
    "courses and training": "training",
    "online courses": "training",
    "professional development": "training",
    "workshops": "training",
    "bootcamp": "training",
    # languages
    "languages": "languages",
    "language skills": "languages",
    "spoken languages": "languages",
    "language proficiency": "languages",
    # awards
    "awards": "awards",
    "achievements": "awards",
    "honors": "awards",
    "honours": "awards",
    "awards & achievements": "awards",
    "recognitions": "awards",
    "accomplishments": "awards",
    # publications
    "publications": "publications",
    "research": "publications",
    "papers": "publications",
    "articles": "publications",
    "conference papers": "publications",
    # additional
    "additional information": "additional_information",
    "additional": "additional_information",
    "activities": "additional_information",
    "activity": "additional_information",
    "extracurricular activities": "additional_information",
    "extracurricular activity": "additional_information",
    "interests": "additional_information",
    "hobbies": "additional_information",
    "references": "additional_information",
    "military service": "military",
    "military": "military",
    "availability": "availability",
    "notice period": "notice_period",
}

# Known human language names (lowercase) → title-cased display form
KNOWN_LANGUAGES: dict[str, str] = {
    "arabic": "Arabic",
    "english": "English",
    "french": "French",
    "german": "German",
    "spanish": "Spanish",
    "italian": "Italian",
    "portuguese": "Portuguese",
    "chinese": "Chinese",
    "mandarin": "Mandarin",
    "japanese": "Japanese",
    "korean": "Korean",
    "russian": "Russian",
    "hindi": "Hindi",
    "urdu": "Urdu",
    "turkish": "Turkish",
    "dutch": "Dutch",
    "swedish": "Swedish",
    "norwegian": "Norwegian",
    "danish": "Danish",
    "finnish": "Finnish",
    "polish": "Polish",
    "czech": "Czech",
    "hungarian": "Hungarian",
    "romanian": "Romanian",
    "greek": "Greek",
    "hebrew": "Hebrew",
    "persian": "Persian",
    "farsi": "Farsi",
    "indonesian": "Indonesian",
    "malay": "Malay",
    "thai": "Thai",
    "vietnamese": "Vietnamese",
    "swahili": "Swahili",
}

LANGUAGE_PROFICIENCY_TOKENS = {
    "native", "fluent", "proficient", "intermediate", "beginner",
    "basic", "advanced", "mother tongue", "bilingual", "conversational",
    "working proficiency", "professional proficiency", "elementary",
    "limited working proficiency", "full professional proficiency",
    "c2", "c1", "b2", "b1", "a2", "a1",
}

# Degree keyword patterns (lowercase fragments)
DEGREE_KEYWORDS = [
    r"\bb\.?sc\.?\b", r"\bm\.?sc\.?\b", r"\bb\.?a\.?\b", r"\bm\.?a\.?\b",
    r"\bb\.?eng\.?\b", r"\bm\.?eng\.?\b", r"\bph\.?d\.?\b", r"\bd\.?phil\.?\b",
    r"\bmba\b", r"\bllb\b", r"\bllm\b", r"\bmd\b", r"\bbds\b",
    r"\bbachelor", r"\bmaster", r"\bdoctor", r"\bassociate",
    r"\bdiploma\b", r"\bcertificate\b", r"\bhigh school\b", r"\bsecondary\b",
]
DEGREE_RE = re.compile("|".join(DEGREE_KEYWORDS), re.IGNORECASE)

# Certification platform hints
CERTIFICATION_KEYWORDS = [
    "certified", "certificate", "certification", "aws", "azure", "google cloud",
    "gcp", "comptia", "cisco", "pmp", "scrum", "agile", "oracle", "microsoft",
    "coursera", "udemy", "edx", "linkedin learning", "professional certificate",
]
CERT_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CERTIFICATION_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Common technology normalization map (lowercase → display form)
TECH_NORMALIZATIONS: dict[str, str] = {
    "node js": "Node.js",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "react js": "React",
    "reactjs": "React",
    "react.js": "React",
    "vue js": "Vue.js",
    "vuejs": "Vue.js",
    "angular js": "AngularJS",
    "angularjs": "AngularJS",
    "next js": "Next.js",
    "nextjs": "Next.js",
    "nuxt js": "Nuxt.js",
    "nuxtjs": "Nuxt.js",
    "mongo db": "MongoDB",
    "mongodb": "MongoDB",
    "typescript": "TypeScript",
    "javascript": "JavaScript",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mysql": "MySQL",
    "graphql": "GraphQL",
    "restful": "RESTful",
    "tailwindcss": "Tailwind CSS",
    "tailwind css": "Tailwind CSS",
    "bootstrap": "Bootstrap",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "docker": "Docker",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "github": "GitHub",
    "gitlab": "GitLab",
    "bitbucket": "Bitbucket",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "scikit-learn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Azure",
    "html5": "HTML5",
    "css3": "CSS3",
}

# Soft skill keywords
SOFT_SKILL_KEYWORDS = {
    "communication", "teamwork", "leadership", "problem solving", "critical thinking",
    "time management", "adaptability", "creativity", "collaboration", "attention to detail",
    "interpersonal", "organizational", "multitasking", "analytical", "flexibility",
    "initiative", "motivated", "proactive", "negotiation", "presentation",
    "conflict resolution", "decision making", "empathy", "mentoring", "coaching",
    "fast learner", "quick learner", "detail oriented", "result oriented",
    "customer oriented", "self motivated", "team player",
}

# Platform domain → candidate type
PLATFORM_DOMAIN_MAP: dict[str, str] = {
    "linkedin.com": "linkedin",
    "github.com": "github",
    "gitlab.com": "github",
    "bitbucket.org": "github",
    "behance.net": "portfolio",
    "dribbble.com": "portfolio",
    "artstation.com": "portfolio",
    "figma.com": "portfolio",
    "notion.so": "website",
    "medium.com": "website",
    "dev.to": "website",
    "stackoverflow.com": "website",
    "kaggle.com": "website",
}

# Military status trigger phrases
MILITARY_TRIGGERS = [
    r"military\s+status[:\s]",
    r"military\s+service[:\s]",
    r"national\s+service[:\s]",
    r"armed\s+forces[:\s]",
    r"military\s+duty[:\s]",
    r"\b(completed|exempt|deferred|serving)\b.{0,30}(military|service|duty)",
]

# Availability patterns
AVAILABILITY_PATTERNS = [
    r"available\s+(immediately|now|asap|from\s+\w+|\w+\s+\d{4})",
    r"availability[:\s]+(.+?)(?:\n|$)",
    r"start\s+date[:\s]+(.+?)(?:\n|$)",
    r"can\s+start\s+(immediately|from\s+\w+|\w+\s+\d{4})",
]

# Notice period patterns
NOTICE_PATTERNS = [
    r"notice\s+period[:\s]+(.+?)(?:\n|$)",
    r"(\d+\s+(?:week|month|day)s?\s+notice)",
    r"notice\s*:\s*(.+?)(?:\n|$)",
]

# Date range patterns for experience/project blocks
DATE_RANGE_RE = re.compile(
    r"""
    (?:
        (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}
        |Q[1-4]\s*\d{4}
        |\d{4}
    )
    \s*(?:–|-|to)\s*
    (?:
        (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}
        |Q[1-4]\s*\d{4}
        |Present|Current|Now|Till\s+date|Till\s+now
        |\d{4}
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

SINGLE_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# GPA pattern
GPA_RE = re.compile(r"\bgpa[:\s]+(\d+\.?\d*)\s*(?:/\s*\d+\.?\d*)?\b", re.IGNORECASE)

# URL pattern
VALID_TLDS = {
    "com", "net", "org", "io", "dev", "ai", "app", "co",
    "me", "info", "biz", "xyz", "tech", "site"
}

BARE_DOMAIN_RE = re.compile(
    r"(?<!@)\b([a-zA-Z0-9][a-zA-Z0-9\-]*)\.([a-zA-Z]{2,})(/[^\s<>\"']*)?\b",
    re.IGNORECASE,
)

PLATFORM_LINK_RE = re.compile(
    r"""
    (?<!@)
    \b
    (?:
        (?:https?://)?(?:www\.)?
        (?:
            linkedin\.com
            |github\.com
            |gitlab\.com
            |bitbucket\.org
            |behance\.net
            |dribbble\.com
            |artstation\.com
            |figma\.com
            |notion\.so
            |medium\.com
            |dev\.to
            |stackoverflow\.com
            |kaggle\.com
        )
        (?:/[^\s<>\"']*)?
        |
        https?://[^\s<>\"']+
        |
        www\.[^\s<>\"']+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

SEPARATOR_LINE_RE = re.compile(r"^\s*[-_=~•·▪▸▹►◆◇○●■□✓✔✗✘─]{3,}\s*$")

# Email pattern
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Phone patterns
PHONE_RE = re.compile(
    r"""
    (?:\+?[\d]{1,3}[\s\-.]?)?          # optional country code
    (?:\([\d]{1,4}\)[\s\-.]?)?         # optional area code in parens
    [\d]{3,5}                           # first group of digits
    [\s\-.]?                            # separator
    [\d]{3,5}                           # second group of digits
    (?:[\s\-.]?[\d]{2,5})?             # optional third group
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """Represents a detected section of a CV."""
    name: str          # Normalized internal name (e.g. "skills", "experience")
    raw_header: str    # The original header text as found in the document
    start_line: int
    end_line: int
    text: str          # Full text of the section (excluding header line)


@dataclass
class Candidate:
    """A single evidence candidate extracted from the CV."""
    candidate_type: str
    value: str
    normalized_value: str
    source_text: str
    section: str
    confidence: float
    extractor: str
    subfields: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "candidate_type": self.candidate_type,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "source_text": self.source_text,
            "section": self.section,
            "confidence": self.confidence,
            "extractor": self.extractor,
        }
        if self.subfields:
            d["subfields"] = self.subfields
        return d


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(raw_text: str) -> str:
    """
    Lightly normalize raw CV text.

    - Decode unicode to NFC form
    - Normalize line endings to \\n
    - Convert common unicode bullets to ASCII dash
    - Collapse repeated blank lines (max 2 consecutive blanks)
    - Collapse intra-line repeated spaces
    - Strip trailing whitespace per line
    - Attempt light repair of split emails and URLs
    """
    # NFC normalize
    text = unicodedata.normalize("NFC", raw_text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize unicode bullets to "-"
    bullet_chars = "•·▪▸▹►◆◇○●■□✓✔✗✘→"
    for ch in bullet_chars:
        text = text.replace(ch, "-")

    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]

    # Collapse runs of more than 2 blank lines into 2
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned.append(line)
        else:
            if SEPARATOR_LINE_RE.fullmatch(line):
                continue
            blank_run = 0
            # Collapse multiple internal spaces
            cleaned.append(re.sub(r" {2,}", " ", line))

    text = "\n".join(cleaned)

    # Light repair: split email (e.g. "john@ example .com" → "john@example.com")
    text = re.sub(r"(\w+)\s*@\s*(\w)", r"\1@\2", text)

    return text


def split_lines(text: str) -> list[str]:
    """Return list of lines from normalized text."""
    return text.split("\n")


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def _is_section_header(line: str) -> Optional[str]:
    """
    Heuristically determine if a line is a CV section header.

    Returns the normalized section name if matched, else None.

    Heuristics:
    - Line must not be too long (headers are typically short labels)
    - Match against known aliases (case-insensitive, stripped of punctuation)
    - Allow lines ending in ':' or '|' as possible headers
    - Reject lines that look like job titles or descriptions
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None

    # Remove trailing punctuation for matching
    candidate_text = re.sub(r"[:\|_\-=]+$", "", stripped).strip().lower()

    # Direct alias lookup
    if candidate_text in SECTION_ALIASES:
        return SECTION_ALIASES[candidate_text]

    # Try partial match at start of line (e.g. "SKILLS & EXPERTISE")
    for alias, normalized in SECTION_ALIASES.items():
        if candidate_text.startswith(alias) or alias.startswith(candidate_text):
            # Only match if close in length
            if abs(len(candidate_text) - len(alias)) <= 8:
                return normalized

    return None


def detect_sections(lines: list[str]) -> list[Section]:
    """
    Detect rough section boundaries in CV lines.

    Returns a list of Section objects ordered by appearance.
    Lines before the first detected section are collected in an "unknown" section
    unless the section boundaries start very early.
    """
    sections: list[Section] = []
    current_header: Optional[str] = None
    current_raw_header: str = ""
    current_start: int = 0
    current_lines: list[str] = []

    def flush_section(end_line: int) -> None:
        nonlocal current_header, current_raw_header, current_start, current_lines
        if current_lines or current_header is not None:
            text = "\n".join(current_lines).strip()
            name = current_header if current_header else "unknown"
            sections.append(
                Section(
                    name=name,
                    raw_header=current_raw_header,
                    start_line=current_start,
                    end_line=end_line,
                    text=text,
                )
            )
        current_header = None
        current_raw_header = ""
        current_start = end_line + 1
        current_lines = []

    for i, line in enumerate(lines):
        detected = _is_section_header(line)
        if detected is not None:
            flush_section(i - 1)
            current_header = detected
            current_raw_header = line.strip()
            current_start = i
        else:
            current_lines.append(line)

    flush_section(len(lines) - 1)
    return sections


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _normalize_tech(term: str) -> str:
    """Apply known tech normalizations to a term."""
    key = term.strip().lower()
    return TECH_NORMALIZATIONS.get(key, term.strip())


def _is_separator_line(line: str) -> bool:
    """Return True for decorative separator lines."""
    return bool(SEPARATOR_LINE_RE.fullmatch(line))


def _extract_labeled_value(line: str, labels: list[str]) -> Optional[str]:
    """Extract the full labeled line when a supported field label is present."""
    stripped = line.strip()
    if not stripped or _is_separator_line(stripped):
        return None
    for label in labels:
        pattern = rf"\b{re.escape(label)}\b\s*[:\-]\s*(.+)"
        match = re.search(pattern, stripped, re.IGNORECASE)
        if match:
            return stripped[match.start():].strip()
    return None


def _is_inline_labeled_field(line: str) -> bool:
    """Return True when a line is an inline labeled field, not a block title."""
    return any(
        extractor(line) is not None
        for extractor in (_extract_military_line, _extract_availability_line, _extract_notice_line)
    )


def _extract_military_line(line: str) -> Optional[str]:
    """Extract a military-status labeled line when present."""
    return _extract_labeled_value(line, ["military status", "military service status", "military service"])


def _extract_availability_line(line: str) -> Optional[str]:
    """Extract an availability labeled line when present."""
    return _extract_labeled_value(line, ["availability", "start date"])


def _extract_notice_line(line: str) -> Optional[str]:
    """Extract a notice-period labeled line when present."""
    return _extract_labeled_value(line, ["notice period", "notice"])


def _classify_url(url: str) -> str:
    """Classify a URL into a candidate type based on domain."""

    url_lower = url.lower()

    # --- NEW: validate TLD ---
    match = BARE_DOMAIN_RE.match(url_lower)
    if match:
        tld = match.group(2)
        if tld not in VALID_TLDS:
            return None  # ❌ reject fake domains like .js

    # --- existing logic ---
    for domain, ctype in PLATFORM_DOMAIN_MAP.items():
        if domain in url_lower:
            return ctype

    if re.match(r"^(?:https?://|www\.)", url_lower):
        return "website"

    return None

def _section_text_for(sections: list[Section], names: list[str]) -> list[Section]:
    """Return sections matching any of the given normalized names."""
    return [s for s in sections if s.name in names]


def _split_skill_items(text: str) -> list[str]:
    """Split a skills line or paragraph into individual items."""
    # Split by common delimiters: comma, semicolon, pipe, bullet dash
    parts = re.split(r"[,;|]|\s{2,}|\n\s*-\s*|\n", text)
    cleaned = []
    for p in parts:
        p = re.sub(r"^[\s\-•*]+", "", p).strip()
        if p and len(p) < 60:
            cleaned.append(p)
    return cleaned


def _is_soft_skill(term: str) -> bool:
    """Return True if term strongly matches a known soft skill."""
    lower = term.lower()
    return any(soft in lower for soft in SOFT_SKILL_KEYWORDS)


def _is_likely_bullet_line(line: str) -> bool:
    """Return True if the line appears to be a bullet-pointed description item."""
    return bool(re.match(r"^\s*[-•*]\s+\S", line))


def _strip_bullet(line: str) -> str:
    """Remove leading bullet characters from a line."""
    return re.sub(r"^[\s\-•*]+", "", line).strip()


def _contains_date_range_like(text: str) -> bool:
    """Return True for supported date-range formats used in CV entries."""
    return bool(
        DATE_RANGE_RE.search(text)
        or re.search(
            r"\b\d{1,2}/\d{4}\s*(?:–|-|to)\s*(?:\d{1,2}/\d{4}|present|current|now)\b",
            text,
            re.IGNORECASE,
        )
    )


def _split_project_title_and_description(line: str) -> tuple[str, str]:
    """
    Split a project line of the form "Project Name: description" when the left
    side looks like a project title rather than a metadata label.
    """
    stripped = line.strip()
    if ":" not in stripped:
        return stripped, ""

    left, right = [part.strip() for part in stripped.split(":", 1)]
    if not left or not right:
        return stripped, ""

    metadata_labels = {
        "tech", "stack", "tools", "technologies", "built with",
        "github", "gitlab", "link", "links", "url", "demo", "role",
    }
    if left.lower() in metadata_labels:
        return stripped, ""

    project_markers = (
        "project", "application", "system", "website", "platform",
        "dashboard", "game", "app", "compiler",
    )
    if len(left) <= 100 and any(marker in left.lower() for marker in project_markers):
        return left, right

    return stripped, ""


def _is_project_entry_start(line: str) -> bool:
    """Return True when a line looks like the start of a project entry."""
    stripped = line.strip()
    if not stripped or _is_likely_bullet_line(stripped) or _is_inline_labeled_field(stripped):
        return False
    if "|" in stripped and _contains_date_range_like(stripped):
        return True
    title, _ = _split_project_title_and_description(stripped)
    if title != stripped:
        return True
    return False


def _is_training_entry_start(line: str) -> bool:
    """Return True when a line looks like the start of a training/course entry."""
    stripped = line.strip()
    if not stripped or _is_likely_bullet_line(stripped) or _is_inline_labeled_field(stripped):
        return False
    training_markers = (
        "training", "course", "courses", "intern", "internship",
        "workshop", "bootcamp", "certificate", "certification",
    )
    lower = stripped.lower()
    has_marker = any(marker in lower for marker in training_markers)
    if "|" in stripped and (_contains_date_range_like(stripped) or has_marker):
        return True
    if _contains_date_range_like(stripped) and has_marker:
        return True
    return bool(
        has_marker
        and len(stripped) <= 120
        and "." not in stripped
        and stripped[:1].isupper()
    )


def _is_activity_entry_start(line: str) -> bool:
    """Return True when a line looks like the start of an activity entry."""
    stripped = line.strip()
    if not stripped or _is_likely_bullet_line(stripped) or _is_inline_labeled_field(stripped):
        return False

    lower = stripped.lower()
    if _contains_date_range_like(stripped):
        return True

    activity_markers = (
        "camp", "club", "society", "chapter", "competition",
        "workshop", "bootcamp", "training", "league", "team",
    )
    return bool(
        len(stripped) <= 120
        and any(marker in lower for marker in activity_markers)
    )


# ---------------------------------------------------------------------------
# Email extraction
# ---------------------------------------------------------------------------

def extract_emails(
    lines: list[str],
    sections: list[Section],
) -> list[Candidate]:
    """Extract email addresses from the CV text."""
    candidates: list[Candidate] = []
    contact_sections = {s.name for s in sections if s.name == "contact"}

    # Determine which lines are in contact/header area (first 15 lines or contact section)
    contact_line_indices: set[int] = set()
    for s in sections:
        if s.name == "contact":
            contact_line_indices.update(range(s.start_line, s.end_line + 1))

    for i, line in enumerate(lines):
        matches = EMAIL_RE.findall(line)
        for m in matches:
            in_contact = i in contact_line_indices or i < 15
            conf = 0.97 if in_contact else 0.90
            section = "contact" if i in contact_line_indices else (
                "unknown" if i < 15 else "body"
            )
            candidates.append(Candidate(
                candidate_type="email",
                value=m,
                normalized_value=m.lower(),
                source_text=line.strip(),
                section=section,
                confidence=conf,
                extractor="email_regex",
            ))

    return candidates


# ---------------------------------------------------------------------------
# Phone extraction
# ---------------------------------------------------------------------------

def extract_phone_numbers(
    lines: list[str],
    sections: list[Section],
) -> list[Candidate]:
    """
    Extract phone numbers using a broad regex.

    Filters out obvious non-phones (pure years, IDs too short, etc.).
    """
    candidates: list[Candidate] = []
    contact_line_indices: set[int] = set()
    for s in sections:
        if s.name == "contact":
            contact_line_indices.update(range(s.start_line, s.end_line + 1))

    seen: set[str] = set()

    for i, line in enumerate(lines):
        # Skip lines that are purely years or date ranges
        if re.fullmatch(r"[\d\s\-./–]{1,6}", line.strip()):
            continue

        for m in PHONE_RE.finditer(line):
            raw = m.group().strip()
            # Minimum viable phone: at least 7 digits
            digit_count = sum(c.isdigit() for c in raw)
            if digit_count < 7 or digit_count > 15:
                continue
            # Skip if it looks like a lone year
            if re.fullmatch(r"(19|20)\d{2}", raw):
                continue
            # Avoid email part captures
            if "@" in line[max(0, m.start() - 2): m.end() + 2]:
                continue

            key = re.sub(r"\D", "", raw)  # digits only for dedup
            if key in seen:
                continue
            seen.add(key)

            in_contact = i in contact_line_indices or i < 15
            conf = 0.92 if in_contact else 0.80
            section = "contact" if i in contact_line_indices else "unknown"

            candidates.append(Candidate(
                candidate_type="phone_number",
                value=raw,
                normalized_value=raw,
                source_text=line.strip(),
                section=section,
                confidence=conf,
                extractor="phone_regex",
            ))

    return candidates


# ---------------------------------------------------------------------------
# Link extraction
# ---------------------------------------------------------------------------

def extract_links(
    lines: list[str],
    sections: list[Section],
) -> list[Candidate]:
    """Detect and classify URLs from CV text."""
    candidates: list[Candidate] = []
    seen_urls: set[str] = set()

    contact_line_indices: set[int] = set()
    for s in sections:
        if s.name == "contact":
            contact_line_indices.update(range(s.start_line, s.end_line + 1))

    for i, line in enumerate(lines):
        for m in PLATFORM_LINK_RE.finditer(line):
            url = m.group().strip().rstrip(".,;)")
            start = m.start()
            end = m.end()
            if (start > 0 and line[start - 1] == "@") or (end < len(line) and line[end:end + 1] == "@"):
                continue
            if url.lower() in seen_urls:
                continue

            ctype = _classify_url(url)

            if not ctype:
                continue

            seen_urls.add(url.lower())

            in_contact = i in contact_line_indices or i < 15
            conf = 0.95 if in_contact else 0.85

            candidates.append(Candidate(
                candidate_type=ctype,
                value=url,
                normalized_value=url,
                source_text=line.strip(),
                section="contact" if i in contact_line_indices else "unknown",
                confidence=conf,
                extractor="url_regex",
            ))

    return candidates


# ---------------------------------------------------------------------------
# Name and job title extraction
# ---------------------------------------------------------------------------

def extract_name_and_job_title(
    lines: list[str],
    sections: list[Section],
) -> list[Candidate]:
    """
    Use top-of-document heuristics to find name and job title candidates.

    Strategy:
    - Scan the first 10 non-blank lines before any recognized section header
    - First candidate: likely the name (short, title-cased, no digits)
    - Second candidate: likely the job title (slightly longer, may contain role keywords)
    - Avoid lines that look like emails, phones, or URLs
    """
    candidates: list[Candidate] = []

    # Identify lines that belong to recognized sections
    section_header_lines: set[int] = {s.start_line for s in sections}

    pre_section_lines: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if i in section_header_lines:
            break
        if i > 20:
            break
        pre_section_lines.append((i, stripped))

    name_found = False
    title_found = False

    for i, (line_no, line) in enumerate(pre_section_lines):
        # Skip if line looks like contact info
        if EMAIL_RE.search(line):
            continue
        if BARE_DOMAIN_RE.search(line):
            continue
        digit_count = sum(c.isdigit() for c in line)
        if digit_count > 3:
            continue
        # Skip long lines (likely descriptions)
        if len(line) > 80:
            continue

        words = line.split()
        if not words:
            continue

        # Name heuristic: short (2-5 words), title-cased or all-caps, no special chars
        if not name_found and 2 <= len(words) <= 5:
            all_title_or_upper = all(
                w[0].isupper() or w.isupper() for w in words if w.isalpha()
            )
            has_only_alpha = all(re.match(r"[A-Za-z'\-]+", w) for w in words)
            if all_title_or_upper and has_only_alpha:
                candidates.append(Candidate(
                    candidate_type="name",
                    value=line,
                    normalized_value=" ".join(w.capitalize() for w in words),
                    source_text=line,
                    section="contact",
                    confidence=0.82,
                    extractor="top_heuristic",
                ))
                name_found = True
                continue

        # Job title heuristic: comes after name, may contain role keywords
        if name_found and not title_found and 1 <= len(words) <= 10:
            role_keywords = {
                "engineer", "developer", "manager", "designer", "analyst",
                "consultant", "architect", "specialist", "lead", "director",
                "officer", "executive", "coordinator", "scientist", "researcher",
                "intern", "associate", "senior", "junior", "principal", "head",
            }
            lower_words = {w.lower() for w in words}
            if lower_words & role_keywords:
                candidates.append(Candidate(
                    candidate_type="job_title",
                    value=line,
                    normalized_value=line,
                    source_text=line,
                    section="contact",
                    confidence=0.80,
                    extractor="top_heuristic",
                ))
                title_found = True

    return candidates


# ---------------------------------------------------------------------------
# Summary extraction
# ---------------------------------------------------------------------------

def extract_summary(sections: list[Section]) -> list[Candidate]:
    """Extract summary/profile/objective text if a matching section exists."""
    candidates: list[Candidate] = []
    for s in sections:
        if s.name == "summary" and s.text.strip():
            # Use first non-empty paragraph
            paragraphs = [p.strip() for p in s.text.split("\n\n") if p.strip()]
            if paragraphs:
                text = paragraphs[0]
                candidates.append(Candidate(
                    candidate_type="summary",
                    value=text,
                    normalized_value=text,
                    source_text=text,
                    section="summary",
                    confidence=0.90,
                    extractor="section_text",
                ))
    return candidates


# ---------------------------------------------------------------------------
# Skills extraction
# ---------------------------------------------------------------------------

def extract_skills(sections: list[Section]) -> list[Candidate]:
    """
    Extract technical and soft skills from skills sections.

    Splits on commas, semicolons, pipes, newlines, and bullets.
    Classifies items as technical_skill or soft_skill.
    """
    candidates: list[Candidate] = []
    skill_sections = _section_text_for(sections, ["skills"])

    for sec in skill_sections:
        lines = sec.text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            items = _split_skill_items(line)
            for item in items:
                if not item or len(item) < 2:
                    continue
                normalized = _normalize_tech(item)
                if _is_soft_skill(item):
                    ctype = "soft_skill"
                    conf = 0.82
                else:
                    ctype = "technical_skill"
                    conf = 0.88
                candidates.append(Candidate(
                    candidate_type=ctype,
                    value=item,
                    normalized_value=normalized,
                    source_text=line,
                    section="skills",
                    confidence=conf,
                    extractor="section_splitter",
                ))

    return candidates


# ---------------------------------------------------------------------------
# Language extraction
# ---------------------------------------------------------------------------

def extract_languages(sections: list[Section]) -> list[Candidate]:
    """
    Extract human language candidates from languages sections.

    Also detects proficiency qualifiers when present.
    """
    candidates: list[Candidate] = []
    lang_sections = _section_text_for(sections, ["languages"])

    for sec in lang_sections:
        lines = sec.text.split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Try to match known language names in the line
            lower_line = stripped.lower()
            for lang_lower, lang_display in KNOWN_LANGUAGES.items():
                if re.search(r"\b" + re.escape(lang_lower) + r"\b", lower_line):
                    # Extract proficiency if present
                    proficiency_found = ""
                    for prof in LANGUAGE_PROFICIENCY_TOKENS:
                        if prof in lower_line:
                            proficiency_found = prof.title()
                            break

                    value = (
                        f"{lang_display} ({proficiency_found})"
                        if proficiency_found
                        else lang_display
                    )
                    candidates.append(Candidate(
                        candidate_type="language",
                        value=value,
                        normalized_value=value,
                        source_text=stripped,
                        section="languages",
                        confidence=0.93,
                        extractor="language_lookup",
                    ))

    return candidates


# ---------------------------------------------------------------------------
# Certifications extraction
# ---------------------------------------------------------------------------

def extract_certifications(sections: list[Section]) -> list[Candidate]:
    """
    Extract certification candidates from certifications sections.

    Each non-blank line in a certifications section is treated as a candidate.
    """
    candidates: list[Candidate] = []
    cert_sections = _section_text_for(sections, ["certifications"])

    for sec in cert_sections:
        lines = sec.text.split("\n")
        for line in lines:
            stripped = _strip_bullet(line)
            if not stripped or len(stripped) < 4:
                continue
            candidates.append(Candidate(
                candidate_type="certification",
                value=stripped,
                normalized_value=stripped,
                source_text=stripped,
                section="certifications",
                confidence=0.90,
                extractor="section_line",
            ))

    return candidates


# ---------------------------------------------------------------------------
# Activities extraction
# ---------------------------------------------------------------------------

def extract_activities(sections: list[Section]) -> list[Candidate]:
    """Extract activity/extracurricular candidates from additional information."""
    candidates: list[Candidate] = []
    info_sections = _section_text_for(sections, ["additional_information"])

    for sec in info_sections:
        lines = [line for line in sec.text.split("\n") if line.strip() and not _is_separator_line(line)]
        blocks: list[list[str]] = []
        current_block: list[str] = []

        for line in lines:
            stripped = line.strip()
            if _is_inline_labeled_field(stripped):
                continue

            starts_new = current_block and _is_activity_entry_start(stripped)
            if starts_new:
                blocks.append(current_block)
                current_block = [stripped]
            else:
                current_block.append(stripped)

        if current_block:
            blocks.append(current_block)

        for block in blocks:
            block_lines = [line.strip() for line in block if line.strip()]
            if not block_lines:
                continue

            title_candidates: list[str] = []
            role_candidates: list[str] = []
            location_candidates: list[str] = []
            date_candidates: list[str] = []
            description_candidates: list[str] = []

            for idx, line in enumerate(block_lines):
                stripped = _strip_bullet(line)
                if not stripped:
                    continue

                if "|" in stripped:
                    parts = [p.strip() for p in stripped.split("|") if p.strip()]
                    for part in parts:
                        if _contains_date_range_like(part):
                            date_candidates.append(part)
                        elif not title_candidates:
                            title_candidates.append(part)
                        elif any(token in part.lower() for token in ("alexandria", "cairo", "egypt", "remote", "online")):
                            location_candidates.append(part)
                        elif not role_candidates:
                            role_candidates.append(part)
                        else:
                            description_candidates.append(part)
                    continue

                if _contains_date_range_like(stripped):
                    date_candidates.append(stripped)
                    continue

                if idx == 0 and not title_candidates:
                    title_candidates.append(stripped)
                    continue

                if any(token in stripped.lower() for token in ("alexandria", "cairo", "egypt", "remote", "online")):
                    location_candidates.append(stripped)
                elif len(stripped) <= 60 and not role_candidates:
                    role_candidates.append(stripped)
                else:
                    description_candidates.append(stripped)

            value_parts = title_candidates + date_candidates
            value = " | ".join(value_parts) if value_parts else block_lines[0]

            candidates.append(Candidate(
                candidate_type="activity",
                value=value,
                normalized_value=value,
                source_text="\n".join(block_lines),
                section="additional_information",
                confidence=0.84,
                extractor="activity_block",
                subfields={
                    "title_candidates": title_candidates,
                    "role_candidates": role_candidates,
                    "location_candidates": location_candidates,
                    "date_candidates": date_candidates,
                    "description_candidates": description_candidates,
                },
            ))

    return candidates


# ---------------------------------------------------------------------------
# Training extraction
# ---------------------------------------------------------------------------

def extract_trainings(sections: list[Section]) -> list[Candidate]:
    """Extract training/course candidates from training sections."""
    candidates: list[Candidate] = []
    training_sections = _section_text_for(sections, ["training"])

    for sec in training_sections:
        blocks = _split_into_blocks(sec.text, mode="training")
        for block in blocks:
            block_lines = [line.strip() for line in block.split("\n") if line.strip()]
            if not block_lines:
                continue

            title_candidates: list[str] = []
            provider_candidates: list[str] = []
            duration_candidates: list[str] = []
            description_candidates: list[str] = []

            for line in block_lines:
                stripped = _strip_bullet(line)
                if not stripped or len(stripped) < 4:
                    continue

                if "|" in stripped:
                    parts = [p.strip() for p in stripped.split("|") if p.strip()]
                    for part in parts:
                        if DATE_RANGE_RE.search(part):
                            duration_candidates.append(part)
                        elif not title_candidates and _is_training_entry_start(part):
                            title_candidates.append(part)
                        elif not title_candidates and _is_training_entry_start(stripped):
                            title_candidates.append(part)
                        elif not provider_candidates:
                            provider_candidates.append(part)
                        else:
                            description_candidates.append(part)
                    continue

                if DATE_RANGE_RE.search(stripped):
                    duration_candidates.append(stripped)
                    continue

                if _is_training_entry_start(stripped) and not title_candidates:
                    title_candidates.append(stripped)
                    continue

                if _is_likely_bullet_line(line) or len(stripped) > 80:
                    description_candidates.append(stripped)
                    continue

                if not title_candidates and not description_candidates:
                    provider_candidates.append(stripped)
                else:
                    description_candidates.append(stripped)

            value_parts = title_candidates + duration_candidates
            value = " | ".join(value_parts) if value_parts else block_lines[0]
            candidates.append(Candidate(
                candidate_type="training",
                value=value,
                normalized_value=value,
                source_text=block,
                section="training",
                confidence=0.88,
                extractor="block_builder",
                subfields={
                    "title_candidates": title_candidates,
                    "provider_candidates": provider_candidates,
                    "duration_candidates": duration_candidates,
                    "description_candidates": description_candidates,
                },
            ))

    return candidates


# ---------------------------------------------------------------------------
# Awards extraction
# ---------------------------------------------------------------------------

def extract_awards(sections: list[Section]) -> list[Candidate]:
    """Extract award and achievement candidates."""
    candidates: list[Candidate] = []
    award_sections = _section_text_for(sections, ["awards"])

    for sec in award_sections:
        lines = sec.text.split("\n")
        for line in lines:
            stripped = _strip_bullet(line)
            if not stripped or len(stripped) < 4:
                continue
            candidates.append(Candidate(
                candidate_type="award",
                value=stripped,
                normalized_value=stripped,
                source_text=stripped,
                section="awards",
                confidence=0.88,
                extractor="section_line",
            ))

    return candidates


# ---------------------------------------------------------------------------
# Publications extraction
# ---------------------------------------------------------------------------

def extract_publications(sections: list[Section]) -> list[Candidate]:
    """Extract publication candidates."""
    candidates: list[Candidate] = []
    pub_sections = _section_text_for(sections, ["publications"])

    for sec in pub_sections:
        # Publications are often multi-line; split on blank lines
        entries = re.split(r"\n\s*\n", sec.text.strip())
        for entry in entries:
            entry = entry.strip()
            if not entry or len(entry) < 8:
                continue
            candidates.append(Candidate(
                candidate_type="publication",
                value=entry,
                normalized_value=entry,
                source_text=entry,
                section="publications",
                confidence=0.88,
                extractor="block_entry",
            ))

    return candidates


# ---------------------------------------------------------------------------
# Military status extraction
# ---------------------------------------------------------------------------

def extract_military_status(
    lines: list[str],
    sections: list[Section],
) -> list[Candidate]:
    """Extract military status only when explicitly stated."""
    candidates: list[Candidate] = []
    for line in lines:
        snippet = _extract_military_line(line)
        if snippet:
            candidates.append(Candidate(
                candidate_type="military_status",
                value=snippet,
                normalized_value=snippet,
                source_text=snippet,
                section="military",
                confidence=0.85,
                extractor="military_heuristic",
            ))

    return candidates


# ---------------------------------------------------------------------------
# Availability extraction
# ---------------------------------------------------------------------------

def extract_availability(lines: list[str]) -> list[Candidate]:
    """Extract availability statement when explicitly stated."""
    candidates: list[Candidate] = []
    for line in lines:
        snippet = _extract_availability_line(line)
        if snippet:
            candidates.append(Candidate(
                candidate_type="availability",
                value=snippet,
                normalized_value=snippet,
                source_text=snippet,
                section="additional_information",
                confidence=0.85,
                extractor="availability_regex",
            ))

    return candidates


# ---------------------------------------------------------------------------
# Notice period extraction
# ---------------------------------------------------------------------------

def extract_notice_period(lines: list[str]) -> list[Candidate]:
    """Extract notice period when explicitly stated."""
    candidates: list[Candidate] = []
    for line in lines:
        snippet = _extract_notice_line(line)
        if snippet:
            candidates.append(Candidate(
                candidate_type="notice_period",
                value=snippet,
                normalized_value=snippet,
                source_text=snippet,
                section="additional_information",
                confidence=0.88,
                extractor="notice_regex",
            ))

    return candidates


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------

def _split_into_blocks(section_text: str, mode: str = "generic") -> list[str]:
    """
    Split a section's text into logical blocks.

    Uses blank lines and date-range anchors as primary block separators.
    """
    if not section_text.strip():
        return []

    if mode == "education":
        lines = [line for line in section_text.strip().split("\n") if line.strip() and not _is_separator_line(line)]
        blocks: list[list[str]] = []
        current_block: list[str] = []
        for line in lines:
            stripped = line.strip()
            if _is_inline_labeled_field(stripped):
                continue
            starts_new = (
                current_block
                and (DATE_RANGE_RE.search(stripped) or SINGLE_YEAR_RE.search(stripped))
                and any(DATE_RANGE_RE.search(existing) or SINGLE_YEAR_RE.search(existing) for existing in current_block)
                and len(current_block) >= 2
            )
            if starts_new:
                blocks.append(current_block)
                current_block = [stripped]
            else:
                current_block.append(stripped)
        if current_block:
            blocks.append(current_block)
        return ["\n".join(block).strip() for block in blocks if any(line.strip() for line in block)]

    if mode == "projects":
        lines = [line for line in section_text.strip().split("\n") if line.strip() and not _is_separator_line(line)]
        blocks: list[list[str]] = []
        current_block: list[str] = []

        for line in lines:
            stripped = line.strip()
            if _is_inline_labeled_field(stripped):
                continue

            starts_new = current_block and _is_project_entry_start(stripped)

            if starts_new:
                blocks.append(current_block)
                current_block = [stripped]
            else:
                current_block.append(stripped)

        if current_block:
            blocks.append(current_block)
        return ["\n".join(block).strip() for block in blocks if any(line.strip() for line in block)]

    if mode == "training":
        lines = [line for line in section_text.strip().split("\n") if line.strip() and not _is_separator_line(line)]
        blocks: list[list[str]] = []
        current_block: list[str] = []
        pending_context: list[str] = []

        for line in lines:
            stripped = line.strip()
            if _is_inline_labeled_field(stripped):
                continue

            if _is_training_entry_start(stripped):
                if current_block:
                    blocks.append(current_block)
                current_block = pending_context + [stripped]
                pending_context = []
                continue

            if current_block:
                current_block.append(stripped)
            else:
                pending_context.append(stripped)

        if current_block:
            blocks.append(current_block)
        elif pending_context:
            blocks.append(pending_context)

        return ["\n".join(block).strip() for block in blocks if any(line.strip() for line in block)]

    # First try splitting on double blank lines
    raw_blocks = re.split(r"\n\s*\n", section_text.strip())

    # If that gives too few blocks (everything in one block) and the text
    # contains multiple date lines, try to re-split on date patterns
    if len(raw_blocks) <= 1:
        # Split on lines that contain a date range
        lines = section_text.strip().split("\n")
        current_block_lines: list[str] = []
        blocks: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or _is_separator_line(stripped):
                continue
            if _is_inline_labeled_field(stripped):
                if current_block_lines:
                    blocks.append("\n".join(current_block_lines))
                    current_block_lines = []
                continue
            if DATE_RANGE_RE.search(stripped) and current_block_lines:
                # New entry starting here
                blocks.append("\n".join(current_block_lines))
                current_block_lines = [stripped]
            else:
                current_block_lines.append(stripped)
        if current_block_lines:
            blocks.append("\n".join(current_block_lines))
        return [b.strip() for b in blocks if b.strip()]

    cleaned_blocks: list[str] = []
    for block in raw_blocks:
        kept_lines = [
            line.strip()
            for line in block.split("\n")
            if line.strip() and not _is_separator_line(line) and not _is_inline_labeled_field(line)
        ]
        if kept_lines:
            cleaned_blocks.append("\n".join(kept_lines))
    return cleaned_blocks


def build_experience_blocks(sections: list[Section]) -> list[Candidate]:
    """
    Build experience_block candidates from experience sections.

    Each block represents one job entry.
    Subfields: title_candidates, company_name_candidates,
               duration_candidates, description_candidates
    """
    candidates: list[Candidate] = []
    exp_sections = _section_text_for(sections, ["experience"])

    for sec in exp_sections:
        blocks = _split_into_blocks(sec.text, mode="experience")

        for block in blocks:
            if not block.strip():
                continue
            block_lines = [l for l in block.split("\n") if l.strip()]
            if not block_lines:
                continue

            title_candidates: list[str] = []
            company_candidates: list[str] = []
            duration_candidates: list[str] = []
            description_candidates: list[str] = []

            for line in block_lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if _is_separator_line(stripped) or _is_inline_labeled_field(stripped):
                    continue

                # Handle pipe-delimited single-line entries:
                # "Title | Company | Jan 2020 - Dec 2021"
                if "|" in stripped and not _is_likely_bullet_line(line):
                    parts_pipe = [p.strip() for p in stripped.split("|") if p.strip()]
                    for part in parts_pipe:
                        if DATE_RANGE_RE.search(part):
                            duration_candidates.append(part)
                        elif not title_candidates:
                            title_candidates.append(part)
                        else:
                            company_candidates.append(part)
                    continue

                # Duration detection
                if DATE_RANGE_RE.search(stripped):
                    duration_candidates.append(stripped)
                    continue

                if _is_likely_bullet_line(line) or len(stripped) > 80:
                    description_candidates.append(_strip_bullet(stripped))
                    continue

                # Heuristic: lines with company-like cues
                company_tokens = {"inc", "ltd", "llc", "co.", "corp", "group", "company",
                                  "technologies", "solutions", "agency", "studio", "labs"}
                lower_stripped = stripped.lower()
                if any(tok in lower_stripped for tok in company_tokens):
                    company_candidates.append(stripped)
                elif not title_candidates:
                    title_candidates.append(stripped)
                elif not company_candidates:
                    company_candidates.append(stripped)
                else:
                    description_candidates.append(stripped)

            # Build compact value string
            parts = title_candidates + company_candidates + duration_candidates
            value = " | ".join(parts) if parts else block_lines[0]

            candidates.append(Candidate(
                candidate_type="experience_block",
                value=value,
                normalized_value=value,
                source_text=block,
                section="experience",
                confidence=0.88,
                extractor="block_builder",
                subfields={
                    "title_candidates": title_candidates,
                    "company_name_candidates": company_candidates,
                    "duration_candidates": duration_candidates,
                    "description_candidates": description_candidates,
                },
            ))

    return candidates


def build_education_blocks(sections: list[Section]) -> list[Candidate]:
    """
    Build education_block candidates from education sections.

    Subfields: institution_candidates, degree_candidates,
               specialization_candidates, graduation_date_candidates,
               gpa_candidates, description_candidates
    """
    candidates: list[Candidate] = []
    edu_sections = _section_text_for(sections, ["education"])

    for sec in edu_sections:
        blocks = _split_into_blocks(sec.text, mode="education")

        for block in blocks:
            block_lines = [l for l in block.split("\n") if l.strip()]
            if not block_lines:
                continue

            institution_candidates: list[str] = []
            degree_candidates: list[str] = []
            specialization_candidates: list[str] = []
            graduation_date_candidates: list[str] = []
            gpa_candidates: list[str] = []
            description_candidates: list[str] = []

            for line in block_lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if _is_separator_line(stripped) or _is_inline_labeled_field(stripped):
                    continue

                # GPA
                if GPA_RE.search(stripped):
                    gpa_candidates.append(stripped)
                    continue

                # Date range → graduation date
                if DATE_RANGE_RE.search(stripped) or SINGLE_YEAR_RE.search(stripped):
                    graduation_date_candidates.append(stripped)
                    # May also contain institution info; let it fall through if short
                    if len(stripped) < 15:
                        continue

                # Degree detection
                if DEGREE_RE.search(stripped):
                    degree_candidates.append(stripped)
                    continue

                # Institution heuristic: contains "university", "college", etc.
                institution_tokens = {
                    "university", "college", "institute", "school", "academy",
                    "polytechnic", "faculty", "campus",
                }
                lower = stripped.lower()
                if any(tok in lower for tok in institution_tokens):
                    institution_candidates.append(stripped)
                    continue

                # Bullets → description
                if _is_likely_bullet_line(line):
                    description_candidates.append(_strip_bullet(stripped))
                    continue

                # Fallback: short lines → specialization, longer → description
                if len(stripped) < 60 and not degree_candidates:
                    specialization_candidates.append(stripped)
                elif len(stripped) < 60:
                    specialization_candidates.append(stripped)
                else:
                    description_candidates.append(stripped)

            parts = degree_candidates + institution_candidates + graduation_date_candidates
            value = " | ".join(parts) if parts else block_lines[0]

            candidates.append(Candidate(
                candidate_type="education_block",
                value=value,
                normalized_value=value,
                source_text=block,
                section="education",
                confidence=0.87,
                extractor="block_builder",
                subfields={
                    "institution_candidates": institution_candidates,
                    "degree_candidates": degree_candidates,
                    "specialization_candidates": specialization_candidates,
                    "graduation_date_candidates": graduation_date_candidates,
                    "gpa_candidates": gpa_candidates,
                    "description_candidates": description_candidates,
                },
            ))

    return candidates


def build_project_blocks(sections: list[Section]) -> list[Candidate]:
    """
    Build project_block candidates from project sections.

    Subfields: project_name_candidates, duration_candidates,
               tool_candidates, description_candidates, link_candidates
    """
    candidates: list[Candidate] = []
    proj_sections = _section_text_for(sections, ["projects"])

    for sec in proj_sections:
        blocks = _split_into_blocks(sec.text, mode="projects")

        for block in blocks:
            block_lines = [l for l in block.split("\n") if l.strip()]
            if not block_lines:
                continue

            project_name_candidates: list[str] = []
            duration_candidates: list[str] = []
            tool_candidates: list[str] = []
            description_candidates: list[str] = []
            link_candidates: list[str] = []

            # Robust URL pattern for project blocks: require http/https/www or known TLD path
            PROJECT_URL_RE = re.compile(
                r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE
            )

            for i, line in enumerate(block_lines):
                stripped = line.strip()
                if not stripped:
                    continue
                if _is_separator_line(stripped) or _is_inline_labeled_field(stripped):
                    continue

                # Tech stack hints: "Tech:", "Stack:", "Built with:", "Technologies:"
                # Check this BEFORE URL detection to avoid "Node.js" being misclassified
                if re.match(r"(?:tech|stack|tools?|technologies|built\s+with)[:\s]+", stripped, re.IGNORECASE):
                    tech_part = re.sub(r"^(?:tech|stack|tools?|technologies|built\s+with)[:\s]+", "", stripped, flags=re.IGNORECASE)
                    items = _split_skill_items(tech_part)
                    tool_candidates.extend(items)
                    continue

                # Link detection (strict: require http/https/www prefix)
                url_matches = PROJECT_URL_RE.findall(stripped)
                if url_matches:
                    link_candidates.extend(url_matches)
                    continue

                if "|" in stripped and not _is_likely_bullet_line(line):
                    parts = [p.strip() for p in stripped.split("|") if p.strip()]
                    for part in parts:
                        if _contains_date_range_like(part):
                            duration_candidates.append(part)
                        elif part.lower() in {"github", "gitlab", "demo"}:
                            continue
                        elif not project_name_candidates:
                            project_name_candidates.append(part)
                        else:
                            tool_candidates.append(part)
                    continue

                # Duration
                if _contains_date_range_like(stripped):
                    duration_candidates.append(stripped)
                    continue

                project_title, inline_description = _split_project_title_and_description(stripped)
                if project_title != stripped:
                    if not project_name_candidates:
                        project_name_candidates.append(project_title)
                    else:
                        description_candidates.append(project_title)
                    if inline_description:
                        description_candidates.append(inline_description)
                    continue

                if _is_likely_bullet_line(line):
                    description_candidates.append(_strip_bullet(stripped))
                    continue

                # First non-blank line is likely the project name
                if i == 0 or (not project_name_candidates and len(stripped) < 80):
                    project_name_candidates.append(stripped)
                    continue

                description_candidates.append(stripped)

            value = project_name_candidates[0] if project_name_candidates else block_lines[0]

            candidates.append(Candidate(
                candidate_type="project_block",
                value=value,
                normalized_value=value,
                source_text=block,
                section="projects",
                confidence=0.85,
                extractor="block_builder",
                subfields={
                    "project_name_candidates": project_name_candidates,
                    "duration_candidates": duration_candidates,
                    "tool_candidates": tool_candidates,
                    "description_candidates": description_candidates,
                    "link_candidates": link_candidates,
                },
            ))

    return candidates


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """
    Conservative deduplication of candidates.

    - Remove exact duplicates (same type + normalized_value)
    - Prefer richer source_text when values match
    - Keep distinct evidence when it is genuinely useful
    - Do not collapse different candidate types
    """
    # Group by (type, normalized_value)
    seen: dict[tuple[str, str], Candidate] = {}

    for cand in candidates:
        key = (cand.candidate_type, cand.normalized_value.strip().lower())
        if key not in seen:
            seen[key] = cand
        else:
            existing = seen[key]
            # Prefer the one with richer source_text
            if len(cand.source_text) > len(existing.source_text):
                seen[key] = cand
            # Prefer higher confidence
            elif cand.confidence > existing.confidence:
                seen[key] = cand

    return list(seen.values())


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_candidates(raw_text: str) -> dict:
    """
    Convert raw CV text into structured evidence candidates.

    Parameters
    ----------
    raw_text : str
        The raw text of the CV (e.g. from a PDF or plain-text file).

    Returns
    -------
    dict
        ``{"candidates": [<candidate_dict>, ...]}``
    """
    # Step 1: Normalize
    text = normalize_text(raw_text)

    # Step 2: Split into lines
    lines = split_lines(text)

    # Step 3: Detect sections
    sections = detect_sections(lines)

    # Step 4: Run all extractors
    all_candidates: list[Candidate] = []

    all_candidates.extend(extract_emails(lines, sections))
    all_candidates.extend(extract_phone_numbers(lines, sections))
    all_candidates.extend(extract_links(lines, sections))
    all_candidates.extend(extract_name_and_job_title(lines, sections))
    all_candidates.extend(extract_summary(sections))
    all_candidates.extend(extract_skills(sections))
    all_candidates.extend(extract_languages(sections))
    all_candidates.extend(extract_certifications(sections))
    all_candidates.extend(extract_activities(sections))
    all_candidates.extend(extract_trainings(sections))
    all_candidates.extend(extract_awards(sections))
    all_candidates.extend(extract_publications(sections))
    all_candidates.extend(extract_military_status(lines, sections))
    all_candidates.extend(extract_availability(lines))
    all_candidates.extend(extract_notice_period(lines))
    all_candidates.extend(build_experience_blocks(sections))
    all_candidates.extend(build_education_blocks(sections))
    all_candidates.extend(build_project_blocks(sections))

    # Step 5: Deduplicate
    deduped = deduplicate_candidates(all_candidates)

    return {"candidates": [c.to_dict() for c in deduped]}
