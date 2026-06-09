import argparse

import matplotlib.pyplot as plt


def parse_loss_data(data_string):
    """
    Parse a string containing training and validation loss data.

    Args:
        data_string (str): String with one record per line in the format:
                           "Epoch <epoch>:<training_loss>:<validation_loss>"

    Returns:
        tuple: A tuple containing lists of epochs, training losses, and validation losses.
    """
    epochs = []
    training_losses = []
    validation_losses = []

    for line in data_string.strip().split("\n"):
        if not line.startswith("Epoch"):
            continue

        parts = line.split(":")
        epochs.append(int(parts[0].split()[1]))
        training_losses.append(float(parts[1].strip()))
        validation_losses.append(float(parts[2].strip()))

    return epochs, training_losses, validation_losses


def plot_losses(data_string, title="Training and Validation Loss", output_file=None):
    """
    Create a plot showing training and validation loss over epochs.

    Args:
        data_string (str): Raw loss data, see parse_loss_data for the format.
        title (str): Plot title.
        output_file (str): If given, save the figure to this path instead of showing it.

    Returns:
        tuple: epochs, training losses, and validation losses.
    """
    epochs, training_losses, validation_losses = parse_loss_data(data_string)
    print(f"Parsed {len(epochs)} epochs.")

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, training_losses, "b-", label="Training Loss", marker="o", linewidth=2)
    plt.plot(epochs, validation_losses, "r-", label="Validation Loss", marker="s", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

    return epochs, training_losses, validation_losses


EXAMPLE_DATA = """Epoch 200:0.370500:0.495938
Epoch 400:0.431300:0.496557
Epoch 600:0.424600:0.484636
Epoch 800:0.512800:0.478931
Epoch 1000:0.529200:0.477183"""


def main():
    parser = argparse.ArgumentParser(description="Plot training and validation loss over epochs")
    parser.add_argument("input_file", nargs="?",
                        help="File with loss data (defaults to built-in example data)")
    parser.add_argument("--output", help="Save the plot to this path instead of showing it")
    parser.add_argument("--title", default="Training and Validation Loss Over Epochs",
                        help="Plot title")

    args = parser.parse_args()

    if args.input_file:
        with open(args.input_file, "r") as f:
            data = f.read()
    else:
        data = EXAMPLE_DATA

    plot_losses(data, title=args.title, output_file=args.output)


if __name__ == "__main__":
    main()
