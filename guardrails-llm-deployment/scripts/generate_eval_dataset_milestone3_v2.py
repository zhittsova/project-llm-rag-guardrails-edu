import argparse
from pathlib import Path

from guardrails_llm.evaluation_dataset import write_evaluation_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the versioned Milestone 3 evaluation dataset."
    )
    parser.add_argument(
        "--replace-frozen-holdout",
        action="store_true",
        help="Explicitly replace a holdout file that differs from the generated version.",
    )
    parser.add_argument(
        "--overwrite-annotations",
        action="store_true",
        help="Explicitly reset the holdout annotation file.",
    )
    args = parser.parse_args()
    output_dir = Path(__file__).resolve().parents[1] / "data"
    manifest = write_evaluation_dataset(
        output_dir,
        replace_frozen_holdout=args.replace_frozen_holdout,
        overwrite_annotations=args.overwrite_annotations,
    )
    print(
        f"wrote {manifest['total_cases']} cases; "
        f"holdout={manifest['holdout_review_status']}"
    )


if __name__ == "__main__":
    main()
