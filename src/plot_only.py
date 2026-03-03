import numpy as np
from utils import plot_rewards

def main():
    # Load previously saved reward and epsilon data
    reward_list = np.load("logs/reward_list.npy")   # # Suppose you saved during training
    epsilon_list = np.load("logs/epsilon_list.npy")

    # Call the plotting function from utils.py
    plot_rewards(reward_list, epsilons=epsilon_list, path="logs/reward_curve.png", window=50)

if __name__ == "__main__":
    main()