import time
import numpy as np
from model import Network

def train_net(net):
    path         = net.path
    hidden_sizes = net.hyperparameters["hidden_sizes"]
    n_epochs     = net.hyperparameters["n_epochs"]
    batch_size   = net.hyperparameters["batch_size"]
    n_it_neg     = net.hyperparameters["n_it_neg"]
    n_it_pos     = net.hyperparameters["n_it_pos"]
    alphas       = net.hyperparameters["alphas"]

    print(f"name = {path}")
    arch = "784-" + "-".join(str(n) for n in hidden_sizes) + "-10"
    print(f"architecture = {arch}")
    print(f"number of epochs = {n_epochs}")
    print(f"batch_size = {batch_size}")
    print(f"n_it_neg = {n_it_neg}")
    print(f"n_it_pos = {n_it_pos}")
    learning_rates_str = " ".join([f"alpha_W{i+1}={alpha:.3f}" for i, alpha in enumerate(alphas)])
    print("learning rates: " + learning_rates_str + "\n")

    n_batches_train = 50000 // batch_size
    n_batches_valid = 10000 // batch_size

    start_time = time.time()

    for epoch in range(n_epochs):

        # --- TRAINING ---
        measures_sum = [0.0, 0.0, 0.0]
        for index in range(n_batches_train):
            net.update_mini_batch_index(index)

            # Negative phase: relax the network.
            net.negative_phase(n_it_neg)

            # Measure energy, cost and error.
            measures = net.measure()
            measures_sum = [ms + m for ms, m in zip(measures_sum, measures)]
            measures_avg = [ms / (index + 1) for ms in measures_sum]
            measures_avg[-1] *= 100.0  # convert error rate to percentage
            print(f"\r{epoch:2d}-train-{(index+1)*batch_size:5d} "
                  f"E={measures_avg[0]:.1f} C={measures_avg[1]:.5f} error={measures_avg[2]:.3f}%", end="")

            # Positive phase: backprop-like relaxation and parameter update.
            net.positive_phase(n_it_pos, *alphas)
        print("")
        net.training_curves["training error"].append(measures_avg[-1])

        # --- VALIDATION ---
        measures_sum = [0.0, 0.0, 0.0]
        for index in range(n_batches_valid):
            net.update_mini_batch_index(n_batches_train + index)
            net.negative_phase(n_it_neg)
            measures = net.measure()
            measures_sum = [ms + m for ms, m in zip(measures_sum, measures)]
            measures_avg = [ms / (index + 1) for ms in measures_sum]
            measures_avg[-1] *= 100.0
            print(f"\r   valid-{(index+1)*batch_size:5d} "
                  f"E={measures_avg[0]:.1f} C={measures_avg[1]:.5f} error={measures_avg[2]:.2f}%", end="")
        print("")

        duration = (time.time() - start_time) / 60.0
        print(f"   duration={duration:.1f} min")

        # SAVE PARAMETERS AT THE END OF THE EPOCH
        net.save_params()

# --- HYPERPARAMETERS ---

speed_net1 = ("speed_net1", {
    "hidden_sizes": [500],
    "n_epochs": 30,
    "batch_size": 20,
    "n_it_neg": 1,
    "n_it_pos": 1,
    "alphas": [np.float32(0.1), np.float32(0.05)]
})

speed_net2 = ("speed_net2", {
    "hidden_sizes": [500, 500],
    "n_epochs": 50,
    "batch_size": 20,
    "n_it_neg": 60,
    "n_it_pos": 1,
    "alphas": [np.float32(0.4), np.float32(0.1), np.float32(0.008)]
})

speed_net3 = ("speed_net3", {
    "hidden_sizes": [500, 500, 500],
    "n_epochs": 150,
    "batch_size": 20,
    "n_it_neg": 400,
    "n_it_pos": 1,
    "alphas": [np.float32(0.4), np.float32(0.1), np.float32(0.015), np.float32(0.002)]
})

if __name__ == "__main__":
    net = Network(*speed_net1)
    train_net(net)
