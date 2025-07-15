import matplotlib.pyplot as plt


def parse_loss_data(data_string):
    """
    Parse a string containing training and validation loss data.

    Args:
        data_string (str): String containing loss data in the format:
                           "Epoch X: Training Loss: Y, Validation Loss: Z"

    Returns:
        tuple: A tuple containing lists of epochs, training losses, and validation losses.
    """
    epochs = []
    training_losses = []
    validation_losses = []

    for line in data_string.strip().split('\n'):
        print(line)
        if line.startswith("Epoch"):
            parts = line.split(':')
            epoch = int(parts[0].split()[1])
            training_loss = float(parts[1].strip())
            validation_loss = float(parts[2].strip())

            epochs.append(epoch)
            training_losses.append(training_loss)
            validation_losses.append(validation_loss)

    return epochs, training_losses, validation_losses


def plot_losses(data_string, title="Training and Validation Loss"):
    """
    Create a plot showing training and validation loss over epochs.
    """
    epochs, training_losses, validation_losses = parse_loss_data(data_string)
    print(f"Parsed data: {len(epochs)} epochs, {len(training_losses)} training losses, {len(validation_losses)} validation losses.")

    plt.figure(figsize=(10, 6))

    # Plot training and validation loss
    plt.plot(epochs, training_losses, 'b-', label='Training Loss', marker='o', linewidth=2)
    plt.plot(epochs, validation_losses, 'r-', label='Validation Loss', marker='s', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Show the plot
    plt.show()

    return epochs, training_losses, validation_losses


if __name__ == "__main__":
    # Example data string

    data = """Epoch 200:0.370500:0.495938
Epoch 400:0.431300:0.496557
Epoch 600:0.424600:0.484636
Epoch 800:0.512800:0.478931
Epoch 1000:0.529200:0.477183"""
    plot_losses(data, title="Training and Validation Loss Over Epochs")
