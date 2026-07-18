import argparse
import json
from pathlib import Path

from guardrails_llm.evaluation_dataset import finalize_holdout_annotations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seal independently reviewed Milestone 3 holdout annotations."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Corpus JSONL used to validate expected_doc_ids.",
    )
    args = parser.parse_args()
    manifest = finalize_holdout_annotations(
        args.data_dir,
        corpus_path=args.corpus,
    )
    print(json.dumps(manifest["annotation_summary"], indent=2))


if __name__ == "__main__":
    main()
