import torch
import os
import matplotlib.pyplot as plt

def save_model(model, path="models/pong_dqn.pth"):
    """
    Save model parameters to the specified path.
    If the path includes a directory, create it automatically.
    """
    dir_name = os.path.dirname(path)
    if dir_name:  # Only create directory if one is specified in the path
        os.makedirs(dir_name, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path="models/pong_dqn.pth"):
    """
    Load model parameters if the file exists.
    If not found, return the untrained model.
    """
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from: {path}")
    else:
        print("Model file not found, using untrained model.")
    return model

def plot_rewards(rewards, epsilons=None, path="logs/reward_curve.png", window=50):
    """
    Plot the reward curve and optional epsilon curve.

    Parameters:
        rewards (list): Total reward per episode.
        epsilons (list, optional): Epsilon values per episode.
        path (str): Path to save the plot.
        window (int): Moving average window size.
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot reward curve
    plt.plot(rewards, label="Episode Reward", color="blue")
    if len(rewards) >= window:
        moving_avg = [sum(rewards[i-window:i]) / window for i in range(window, len(rewards)+1)]
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f"Moving Avg ({window})", color="orange")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward", color="blue")
    plt.title("Training Reward & Epsilon Curve")
    plt.grid(True)

    # Plot epsilon curve on secondary y-axis if provided
    if epsilons is not None:
        ax2 = plt.gca().twinx()
        ax2.plot(epsilons, color="green", linestyle="--", label="Epsilon")
        ax2.set_ylabel("Epsilon", color="green")

        # Merge legends and remove duplicates
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2

        unique = dict(zip(labels, lines))  # Ensure each label appears only once
        plt.legend(unique.values(), unique.keys(), loc="upper right")
    else:
        plt.legend(loc="upper right")

    plt.savefig(path)
    plt.close()
    print(f"Reward and epsilon curve saved to {path}")