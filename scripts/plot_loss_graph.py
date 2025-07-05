import matplotlib.pyplot as plt
import pandas as pd

def plot_loss_graph(file_path):
    df = pd.read_csv(file_path)
    plt.plot(df["step"], df["loss"])
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.show()