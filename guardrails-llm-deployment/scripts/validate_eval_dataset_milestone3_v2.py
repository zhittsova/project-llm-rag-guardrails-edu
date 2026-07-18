import argparse
import json
from pathlib import Path

from guardrails_llm.evaluation_dataset import load_and_validate_evaluation_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the versioned Milestone 3 evaluation dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument(
        "--require-reviewed-holdout",
        action="store_true",
        help="Fail unless every frozen holdout case has two labels and an adjudication.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Corpus JSONL used to validate expected_doc_ids.",
    )
    args = parser.parse_args()
    summary = load_and_validate_evaluation_dataset(
        args.data_dir,
        require_reviewed_holdout=args.require_reviewed_holdout,
        corpus_path=args.corpus,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
