from __future__ import annotations

import argparse
from pathlib import Path

from guardrails_llm.eval_case_annotation import write_labeled_eval_cases


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write an explicit, versioned Milestone 3 evaluation set."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "eval_cases_milestone3.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "eval_cases_milestone3_labeled_v1.jsonl",
    )
    parser.add_argument(
        "--canonical-labels",
        type=Path,
        default=ROOT / "data" / "eval_cases_milestone3_holdout_v3.jsonl",
    )
    args = parser.parse_args()

    try:
        count = write_labeled_eval_cases(
            args.input,
            args.output,
            args.canonical_labels,
        )
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Wrote {count} labeled cases to {args.output}")


if __name__ == "__main__":
    main()
