"""
ce.py
=====
CV extraction pipeline.

Stage 1 — Extract:   Run rule-based extractor on raw CV text → write extracted_results.jsonl
Stage 2 — Enrich:    Send extracted results to a model API (stubbed → reads from file) →
                      receive final candidates → merge with cv_parser_output via schema_enricher
Stage 3 — Output:    Write enriched results to cv_enriched_results.jsonl
"""

import json
from pathlib import Path

from extractor.cv_extractor import extract_candidates
from schema_enricher import enrich_schema


# ============================================================
# Paths
# ============================================================

DATA_PATH            = Path(__file__).with_name("cv_parser_output.jsonl")
EXTRACTED_PATH       = Path(__file__).with_name("cv_extracted_results.jsonl")
MODEL_RESPONSE_PATH  = Path(__file__).with_name("cv_model_response.jsonl")   # stub input
ENRICHED_PATH        = Path(__file__).with_name("cv_enriched_results.jsonl")

NUM_CVS_TO_WRITE = 10
START_INDEX      = 0


# ============================================================
# Stage 1 — Extract candidates and write to file
# ============================================================

def write_extracted_results(
    input_path: Path,
    output_path: Path,
    num_cvs: int,
    start_index: int = 0,
) -> int:
    """
    Run the rule-based extractor on each CV and write results to output_path.

    Returns the number of CVs written.
    """
    written = 0

    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:

        for idx, line in enumerate(infile):
            if idx < start_index:
                continue
            if written >= num_cvs:
                break

            payload = json.loads(line)
            raw_text = payload.get("raw_text", "").strip()

            if not raw_text:
                continue

            output_row = {
                "cv_id": payload.get("cv_id"),
                "extracted_results": extract_candidates(raw_text),
            }

            outfile.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            written += 1

    print(f"[Stage 1] Wrote {written} CV(s) to {output_path}")
    return written


# ============================================================
# Stage 2 — Model API (send / receive)
# ============================================================

def send_to_model_api(extracted_path: Path) -> None:
    """
    TODO: Replace this stub with a real model API call.

    Expected behaviour when implemented:
    - Read extracted_path (cv_extracted_results.jsonl)
    - POST each row (or a batch) to the model endpoint
    - Write the model's returned candidates to MODEL_RESPONSE_PATH
      using the same JSONL schema:
          {"cv_id": "...", "model_candidates": [...]}

    For now this is a no-op; results are read directly from
    MODEL_RESPONSE_PATH (a pre-existing file you supply).
    """
    print(
        f"[Stage 2] send_to_model_api() is a stub — "
        f"reading model results directly from {MODEL_RESPONSE_PATH}"
    )


def load_model_responses(model_response_path: Path) -> dict[str, list]:
    """
    Load model-returned candidates from a JSONL file.

    Each line must be:
        {"cv_id": "...", "model_candidates": [...]}

    Returns a dict keyed by cv_id → list of candidate dicts.
    """
    responses: dict[str, list] = {}

    if not model_response_path.exists():
        print(
            f"[Stage 2] Warning: model response file not found at "
            f"{model_response_path}. Enrichment will proceed with no model candidates."
        )
        return responses

    with model_response_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cv_id = row.get("cv_id")
            if cv_id is not None:
                responses[str(cv_id)] = row.get("model_candidates", [])

    print(f"[Stage 2] Loaded model responses for {len(responses)} CV(s).")
    return responses


# ============================================================
# Stage 3 — Enrich and write final output
# ============================================================

def write_enriched_results(
    input_path: Path,
    extracted_path: Path,
    model_responses: dict[str, list],
    output_path: Path,
    num_cvs: int,
    start_index: int = 0,
) -> int:
    """
    Merge cv_parser_output with rule-based candidates + model candidates
    using schema_enricher, then write to output_path.

    Parameters
    ----------
    input_path      : original cv_parser_output.jsonl (provides the base parsed_output)
    extracted_path  : cv_extracted_results.jsonl (rule-based candidates)
    model_responses : dict of cv_id → model candidate list
    output_path     : destination file for enriched results
    """
    # Index rule-based candidates by cv_id for O(1) lookup
    rule_based: dict[str, list] = {}
    with extracted_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cv_id = str(row["cv_id"])
            rule_based[cv_id] = row["extracted_results"].get("candidates", [])

    written = 0

    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:

        for idx, line in enumerate(infile):
            if idx < start_index:
                continue
            if written >= num_cvs:
                break

            payload = json.loads(line)
            cv_id   = str(payload.get("cv_id"))

            # Base parsed output (everything the parser already knows)
            parsed_output = {
                k: v for k, v in payload.items()
                if k not in ("raw_text", "cv_id")
            }

            # Merge candidates: rule-based first, then model
            all_candidates = (
                rule_based.get(cv_id, [])
                + model_responses.get(cv_id, [])
            )

            enriched = enrich_schema(parsed_output, all_candidates)

            outfile.write(
                json.dumps(
                    {"cv_id": payload.get("cv_id"), "enriched": enriched},
                    ensure_ascii=False,
                ) + "\n"
            )
            written += 1

    print(f"[Stage 3] Wrote {written} enriched CV(s) to {output_path}")
    return written


# ============================================================
# Pipeline entry point
# ============================================================

def run_pipeline(
    num_cvs: int    = NUM_CVS_TO_WRITE,
    start_index: int = START_INDEX,
) -> None:
    # --- Stage 1: Extract ---
    write_extracted_results(
        input_path=DATA_PATH,
        output_path=EXTRACTED_PATH,
        num_cvs=num_cvs,
        start_index=start_index,
    )

    # --- Stage 2: (Stub) Send → Receive model candidates ---
    send_to_model_api(EXTRACTED_PATH)
    model_responses = load_model_responses(MODEL_RESPONSE_PATH)

    # --- Stage 3: Enrich + write final output ---
    write_enriched_results(
        input_path=DATA_PATH,
        extracted_path=EXTRACTED_PATH,
        model_responses=model_responses,
        output_path=ENRICHED_PATH,
        num_cvs=num_cvs,
        start_index=start_index,
    )

    print("[Pipeline] Done.")


if __name__ == "__main__":
    run_pipeline()