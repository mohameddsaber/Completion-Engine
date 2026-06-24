# schema_enricher.py

from copy import deepcopy


# ============================================================
# Helpers
# ============================================================

def first(lst):
    return lst[0] if lst else None


def append_unique(container, key, value):
    container.setdefault(key, [])

    if value not in container[key]:
        container[key].append(value)


def normalize_key(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# ============================================================
# Structured Block Handlers
# ============================================================

def handle_experience(result, candidate):
    sub = candidate.get("subfields", {})

    result.setdefault("experience", [])

    result["experience"].append({
        "title": first(sub.get("title_candidates", [])),
        "company": first(sub.get("company_name_candidates", [])),
        "duration": first(sub.get("duration_candidates", [])),
        "description": "\n".join(
            sub.get("description_candidates", [])
        ),
        "source_text": candidate.get("source_text")
    })


def handle_education(result, candidate):
    sub = candidate.get("subfields", {})

    result.setdefault("education", [])

    result["education"].append({
        "institution": first(
            sub.get("institution_candidates", [])
        ),
        "degree": first(
            sub.get("degree_candidates", [])
        ),
        "specialization": first(
            sub.get("specialization_candidates", [])
        ),
        "graduation_date": first(
            sub.get("graduation_date_candidates", [])
        ),
        "gpa": first(
            sub.get("gpa_candidates", [])
        ),
        "description": "\n".join(
            sub.get("description_candidates", [])
        )
    })


def handle_project(result, candidate):
    sub = candidate.get("subfields", {})

    result.setdefault("projects", [])

    result["projects"].append({
        "name": first(
            sub.get("project_name_candidates", [])
        ),
        "duration": first(
            sub.get("duration_candidates", [])
        ),
        "tools": sub.get(
            "tool_candidates", []
        ),
        "description": "\n".join(
            sub.get("description_candidates", [])
        ),
        "links": sub.get(
            "link_candidates", []
        )
    })


def handle_training(result, candidate):
    sub = candidate.get("subfields", {})

    result.setdefault("trainings", [])

    result["trainings"].append({
        "title": first(
            sub.get("title_candidates", [])
        ),
        "provider": first(
            sub.get("provider_candidates", [])
        ),
        "duration": first(
            sub.get("duration_candidates", [])
        ),
        "description": "\n".join(
            sub.get("description_candidates", [])
        )
    })


def handle_activity(result, candidate):
    sub = candidate.get("subfields", {})

    result.setdefault("activities", [])

    result["activities"].append({
        "title": first(
            sub.get("title_candidates", [])
        ),
        "role": first(
            sub.get("role_candidates", [])
        ),
        "location": first(
            sub.get("location_candidates", [])
        ),
        "date": first(
            sub.get("date_candidates", [])
        ),
        "description": "\n".join(
            sub.get("description_candidates", [])
        )
    })


BLOCK_HANDLERS = {
    "experience_block": handle_experience,
    "education_block": handle_education,
    "project_block": handle_project,
    "training": handle_training,
    "activity": handle_activity,
}


# ============================================================
# Explicit Single Value Fields
# ============================================================

SINGLE_VALUE_FIELDS = {
    "email",
    "phone_number",
    "linkedin",
    "github",
    "website",
    "name",
    "job_title",
    "summary",
    "availability",
    "notice_period",
    "military_status",
}


# ============================================================
# Explicit List Fields
# ============================================================

LIST_FIELDS = {
    "language": "languages",
    "technical_skill": "technical_skills",
    "soft_skill": "soft_skills",
    "certification": "certifications",
    "award": "awards",
    "publication": "publications",
}


# ============================================================
# Unknown Candidate Handler
# ============================================================

def handle_unknown(result, candidate):
    """
    Automatically handles future candidate types.

    Examples:

    patent
    hackathon
    volunteering
    competition
    scholarship
    research_project

    without any code changes.
    """

    section_name = (
        candidate.get("section")
        or candidate.get("candidate_type")
        or "miscellaneous"
    )

    section_name = normalize_key(section_name)

    subfields = candidate.get("subfields", {})

    # -------------------------
    # Structured candidate
    # -------------------------
    if subfields:

        structured = {}

        for key, value in subfields.items():

            if isinstance(value, list):

                if len(value) == 1:
                    structured[key] = value[0]
                else:
                    structured[key] = value

            else:
                structured[key] = value

        structured["_value"] = candidate.get("value")
        structured["_confidence"] = candidate.get("confidence")

        result.setdefault(section_name, [])
        result[section_name].append(structured)

        return

    # -------------------------
    # Single primitive value
    # -------------------------
    value = candidate.get("value")

    if section_name not in result:
        result[section_name] = value
        return

    # -------------------------
    # Convert existing value
    # into list if necessary
    # -------------------------
    if not isinstance(result[section_name], list):

        result[section_name] = [
            result[section_name]
        ]

    if value not in result[section_name]:
        result[section_name].append(value)


# ============================================================
# Main Candidate Application
# ============================================================

def apply_candidate(result, candidate):

    ctype = candidate["candidate_type"]

    # ------------------------------------
    # Structured blocks
    # ------------------------------------
    if ctype in BLOCK_HANDLERS:
        BLOCK_HANDLERS[ctype](result, candidate)
        return

    # ------------------------------------
    # Explicit single-value fields
    # ------------------------------------
    if ctype in SINGLE_VALUE_FIELDS:

        if not result.get(ctype):
            result[ctype] = candidate["value"]

        return

    # ------------------------------------
    # Explicit list fields
    # ------------------------------------
    if ctype in LIST_FIELDS:

        field = LIST_FIELDS[ctype]

        append_unique(
            result,
            field,
            candidate["value"]
        )

        return

    # ------------------------------------
    # Future candidate types
    # ------------------------------------
    handle_unknown(result, candidate)


# ============================================================
# Public API
# ============================================================

def enrich_schema(parsed_output, candidates):

    result = deepcopy(parsed_output)

    for candidate in candidates:
        apply_candidate(result, candidate)

    return result