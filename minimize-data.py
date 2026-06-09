import pandas as pd
import argparse
import sys

def sample_tsv_file(input_file, output_file, keep_percentage=0.2, preserve_header=True):
    """
    Randomly sample a percentage of rows from a TSV file.

    Args:
        input_file (str): Path to input TSV file
        output_file (str): Path to output TSV file
        keep_percentage (float): Percentage of data to keep (0.2 = keep 20%, remove 80%)
        preserve_header (bool): Whether to preserve the header row
    """
    try:
        print(f"Reading TSV file: {input_file}")

        # Read the TSV file
        df = pd.read_csv(input_file, sep='\t')

        original_rows = len(df)
        print(f"Original file has {original_rows} rows")

        # Sample the data
        sampled_df = df.sample(frac=keep_percentage, random_state=42)

        sampled_rows = len(sampled_df)
        removed_rows = original_rows - sampled_rows

        print(f"Keeping {sampled_rows} rows ({keep_percentage*100:.1f}%)")
        print(f"Removing {removed_rows} rows ({(1-keep_percentage)*100:.1f}%)")

        # Save the sampled data
        sampled_df.to_csv(output_file, sep='\t', index=False)

        print(f"Sampled data saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Randomly sample data from a TSV file')
    parser.add_argument('input_file', help='Input TSV file path')
    parser.add_argument('output_file', help='Output TSV file path')
    parser.add_argument('--keep', type=float, default=0.2,
                        help='Percentage of data to keep (default: 0.2 for 20%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Validate keep percentage
    if not 0 < args.keep <= 1:
        print("Error: --keep must be between 0 and 1")
        sys.exit(1)

    # Set random seed for reproducibility
    if args.seed:
        import random
        random.seed(args.seed)

    sample_tsv_file(args.input_file, args.output_file, args.keep)

if __name__ == "__main__":
    main()