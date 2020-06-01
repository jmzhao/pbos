"""
Script used to plot loss
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log(log_path):
    epochs, loss, epoch_times, total_times = [], [], [], []
    with open(log_path, 'r') as f:
        for line in f:
            if not line.startswith("INFO:pbos_train:epoch") or "time" not in line:
                continue
            parts = line.split()
            epochs.append(int(parts[1]))
            loss.append(float(parts[6]))
            epoch_times.append(float(parts[9][:-1]))
            total_times.append(float(parts[11][:-1]))
    return epochs, loss, epoch_times, total_times


def plot_loss(result_paths):
    for model_path in Path(result_paths).iterdir():
        log_path = model_path / "info.log"
        if not log_path.exists():
            continue
        epochs, loss, epoch_times, total_times = parse_log(log_path)
        plt.plot(epochs, loss, '.')
        plt.title(str(model_path))
        plt.show()

        print(f"average epoch time for {str(model_path):<60}: {total_times[-1] / epochs[-1]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', help="path to the results directory", default="results/ws_affix")
    args = parser.parse_args()

    plot_loss(args.results_dir)