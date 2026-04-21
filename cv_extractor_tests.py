import json
from pathlib import Path
from cv_extractor import extract_candidates

DATA_PATH = Path(__file__).with_name("cv_parser_output.jsonl")
OUTPUT_PATH = Path(__file__).with_name("cv_extracted_results.jsonl")

NUM_CVS_TO_WRITE = 10
START_INDEX = 0


def write_extracted_results(
    input_path: Path,
    output_path: Path,
    num_cvs: int,
    start_index: int = 0,
) -> None:
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

    print(f"Wrote {written} CV(s) to {output_path}")


if __name__ == "__main__":
    write_extracted_results(
        input_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        num_cvs=NUM_CVS_TO_WRITE,
        start_index=START_INDEX,
    )