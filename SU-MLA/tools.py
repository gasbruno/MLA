import numpy as np
import matplotlib.pyplot as plt

# Visualisation des résultats
def plot_decision_boundary(model):
    # cycle over 2D grid
    plt.figure(figsize=(5, 5))
    for i in np.arange(-0.1, 1.1, 0.05):
        for j in np.arange(-0.1, 1.1, 0.05):
            # eval model on each grid point
            input_data = torch.tensor([[i, j]], dtype=torch.float32)
            output = model(input_data)
            if output > 0.5:
                plt.plot(i, j, ".r")
            else:
                plt.plot(i, j, ".b")

    for i in range(Y.size(0)):
        if Y[i] == 1:
            plt.plot(X[i, 0], X[i, 1], "ro")
        else:
            plt.plot(X[i, 0], X[i, 1], "bo")
    plt.show()
    