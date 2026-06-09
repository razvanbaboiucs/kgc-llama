import argparse
import sys

import pandas as pd


def sample_tsv_file(input_file, output_file, keep_percentage=0.2, seed=42):
    """
    Randomly sample a percentage of rows from a TSV file.

    Args:
        input_file (str): Path to input TSV file
        output_file (str): Path to output TSV file
        keep_percentage (float): Percentage of data to keep (0.2 = keep 20%, remove 80%)
        seed (int): Random seed for reproducible sampling
    """
    print(f"Reading TSV file: {input_file}")
    df = pd.read_csv(input_file, sep="\t")

    original_rows = len(df)
    print(f"Original file has {original_rows} rows")

    sampled_df = df.sample(frac=keep_percentage, random_state=seed)

    sampled_rows = len(sampled_df)
    removed_rows = original_rows - sampled_rows
    print(f"Keeping {sampled_rows} rows ({keep_percentage * 100:.1f}%)")
    print(f"Removing {removed_rows} rows ({(1 - keep_percentage) * 100:.1f}%)")

    sampled_df.to_csv(output_file, sep="\t", index=False)
    print(f"Sampled data saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Randomly sample data from a TSV file")
    parser.add_argument("input_file", help="Input TSV file path")
    parser.add_argument("output_file", help="Output TSV file path")
    parser.add_argument("--keep", type=float, default=0.2,
                        help="Percentage of data to keep (default: 0.2 for 20%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    if not 0 < args.keep <= 1:
        parser.error("--keep must be between 0 and 1")

    try:
        sample_tsv_file(args.input_file, args.output_file, args.keep, args.seed)
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
