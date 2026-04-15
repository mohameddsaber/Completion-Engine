"""
cv_extractor_tests.py
=====================
Example usage and test cases for the cv_extractor module.

Run with:
    python cv_extractor_tests.py
"""

import json
import sys
from pathlib import Path
from cv_extractor import extract_candidates


DATA_PATH = Path(__file__).with_name("cv_parser_output.jsonl")


def load_raw_text_rows(limit: int = 10) -> list[str]:
    """Load raw CV text values from the JSONL fixture file."""
    rows: list[str] = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            payload = json.loads(line)
            raw_text = payload.get("raw_text", "")
            if raw_text.strip():
                rows.append(raw_text)
    if len(rows) < limit:
        raise AssertionError(f"Expected at least {limit} raw_text rows in {DATA_PATH}")
    return rows


RAW_TEXT_ROWS = load_raw_text_rows(110)
SIMPLE_CV = RAW_TEXT_ROWS[5]


def pick_raw_text(predicate) -> str:
    """Return the first raw_text row that matches the provided predicate."""
    for raw_text in RAW_TEXT_ROWS:
        if predicate(raw_text):
            return raw_text
    raise AssertionError("No matching raw_text row found in cv_parser_output.jsonl")


def pick_raw_text_by_candidates(predicate) -> str:
    """Return the first raw_text row whose extracted candidates match the predicate."""
    for raw_text in RAW_TEXT_ROWS:
        candidates = extract_candidates(raw_text)["candidates"]
        if predicate(candidates):
            return raw_text
    raise AssertionError("No matching extracted-candidate row found in cv_parser_output.jsonl")

print("=" * 60)
print("EXAMPLE USAGE OUTPUT")
print("=" * 60)
result = extract_candidates(SIMPLE_CV)
print(json.dumps(result, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def assert_has_type(candidates: list[dict], ctype: str, min_count: int = 1) -> None:
    """Assert at least min_count candidates of ctype are present."""
    matches = [c for c in candidates if c["candidate_type"] == ctype]
    assert len(matches) >= min_count, (
        f"Expected at least {min_count} candidates of type '{ctype}', got {len(matches)}.\n"
        f"All types present: {sorted(set(c['candidate_type'] for c in candidates))}"
    )


def assert_value_present(candidates: list[dict], ctype: str, substring: str) -> None:
    """Assert at least one candidate of ctype contains substring in value."""
    matches = [
        c for c in candidates
        if c["candidate_type"] == ctype and substring.lower() in c["value"].lower()
    ]
    assert matches, (
        f"No candidate of type '{ctype}' contains '{substring}'.\n"
        f"Candidates of this type: {[c['value'] for c in candidates if c['candidate_type'] == ctype]}"
    )


def assert_subfield_present(
    candidates: list[dict], ctype: str, subfield: str, substring: str
) -> None:
    """Assert at least one block candidate of ctype has substring in subfield list."""
    matches = [
        c for c in candidates
        if c["candidate_type"] == ctype and subfield in c.get("subfields", {})
        and any(substring.lower() in v.lower() for v in c["subfields"][subfield])
    ]
    assert matches, (
        f"No candidate of type '{ctype}' has '{substring}' in subfield '{subfield}'."
    )


# ---------------------------------------------------------------------------
# Test 1: Contact info extraction (email, phone, LinkedIn, GitHub)
# ---------------------------------------------------------------------------

TEST_1_CV = pick_raw_text(
    lambda text: "linkedin.com" in text.lower() and "github.com" in text.lower() and "@" in text
)

def test_contact_info():
    result = extract_candidates(TEST_1_CV)
    cands = result["candidates"]

    assert_has_type(cands, "email")
    assert_value_present(cands, "email", "@")

    assert_has_type(cands, "phone_number")
    assert_has_type(cands, "phone_number")

    assert_has_type(cands, "linkedin")
    assert_has_type(cands, "linkedin")

    assert_has_type(cands, "github")
    assert_has_type(cands, "github")

    print("[PASS] test_contact_info")


# ---------------------------------------------------------------------------
# Test 2: Skills, languages, certifications
# ---------------------------------------------------------------------------

TEST_2_CV = pick_raw_text(
    lambda text: "certif" in text.lower()
)
TEST_2_FIELD_CV = pick_raw_text(lambda text: "military" in text.lower())

def test_skills_languages_certifications():
    result = extract_candidates(TEST_2_CV)
    cands = result["candidates"]
    field_result = extract_candidates(TEST_2_FIELD_CV)
    field_cands = field_result["candidates"]

    # Technical skills
    assert_has_type(cands, "technical_skill", min_count=3)
    assert_value_present(cands, "technical_skill", "Docker")

    # Military
    assert_has_type(field_cands, "military_status")

    print("[PASS] test_skills_languages_certifications")


# ---------------------------------------------------------------------------
# Test 3: Block candidates (experience, education, project blocks)
# ---------------------------------------------------------------------------

TEST_3_CV = pick_raw_text_by_candidates(
    lambda cands: (
        sum(c["candidate_type"] == "experience_block" for c in cands) >= 1
        and sum(c["candidate_type"] == "education_block" for c in cands) >= 1
        and sum(c["candidate_type"] == "project_block" for c in cands) >= 1
    )
)

TEST_4_CV = pick_raw_text(
    lambda text: "ACM Algorithms Training Camp" in text and "ICT and CS Assistant Teacher" in text
)

def test_block_candidates():
    result = extract_candidates(TEST_3_CV)
    cands = result["candidates"]

    # Experience blocks
    assert_has_type(cands, "experience_block", min_count=1)
    assert_subfield_present(cands, "experience_block", "title_candidates", "")

    # Education blocks
    assert_has_type(cands, "education_block", min_count=1)
    assert_subfield_present(cands, "education_block", "degree_candidates", "")
    assert_subfield_present(cands, "education_block", "institution_candidates", "")

    # Project blocks
    assert_has_type(cands, "project_block", min_count=1)
    assert_subfield_present(cands, "project_block", "project_name_candidates", "Project")

    print("[PASS] test_block_candidates")


# ---------------------------------------------------------------------------
# Test 4: Activity candidates
# ---------------------------------------------------------------------------

def test_activity_candidates():
    result = extract_candidates(TEST_4_CV)
    cands = result["candidates"]

    assert_has_type(cands, "activity", min_count=2)
    assert_value_present(cands, "activity", "ACM Algorithms Training Camp")
    assert_value_present(cands, "activity", "ICT and CS Assistant Teacher")

    print("[PASS] test_activity_candidates")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)

    failed = 0
    for test_fn in [
        test_contact_info,
        test_skills_languages_certifications,
        test_block_candidates,
        test_activity_candidates,
    ]:
        try:
            test_fn()
        except AssertionError as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test_fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print()
    if failed == 0:
        print("All tests passed.")
    else:
        print(f"{failed} test(s) failed.")
        sys.exit(1)
