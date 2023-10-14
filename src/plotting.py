import numpy as np
import matplotlib.pyplot as plt


def plot_saliency_eps(model, idx, epsilons, saliency_dict, predicted_label_dict, true_labels):
    heatmap_eps = saliency_dict[model][:, idx, :]
    predicted_labels_i = predicted_label_dict[model][:, idx]
    fig, ax = plt.subplots(1, 4, sharex=True, figsize=(16, 9), dpi=72)

    plt.suptitle(f"Test example {idx}: saliency vs epsilon")

    x = np.array(epsilons)
    y = np.arange(24)
    X, Y = np.meshgrid(x, y)
    Z = heatmap_eps.T

    ax[0].pcolor(X, Y, Z, shading="nearest")

    #ax[0].imshow(heatmap_eps.T)
    ax[0].set_ylabel("Hour")
    ax[0].set_xlabel("Epsilon")
    #ax[0].set_aspect("auto")

    ax[1].plot(epsilons, heatmap_eps.mean(axis=-1))
    ax[1].set_ylabel("Mean saliency")
    ax[1].set_xlabel("Epsilon")

    ax[2].plot(epsilons, heatmap_eps.max(axis=-1))
    ax[2].set_ylabel("Max saliency")
    ax[2].set_xlabel("Epsilon")
    
    ax[3].plot(epsilons, predicted_labels_i, label='Predict')
    ax[3].axhline(y=true_labels[idx], c='k', ls='--', label='True')
    ax[3].set_ylabel("Label")
    ax[3].set_xlabel("Epsilon")
    ax[3].legend()

    fig.tight_layout()
    plt.show()
    #fig.canvas.draw()
    #print("Finished")


def plot_saliency_heatmap(sal, y_true, idx_test_sorted=None, ax=None):
    if idx_test_sorted is None:
        idx_test_sorted = np.argsort(y_true)
    
    if ax is None:
        fig, (ax_main, ax_class) = plt.subplots(1, 2, sharey=True)
        ax_class.set_title("Class label")
        ax_class.imshow(y_true[idx_test_sorted, None], aspect='0.01', cmap='inferno')
        ax_class.set_xticks([])
        ax_main.set_title("Saliency on test set")
    else:
        ax_main = ax
    
    ax_main.imshow((sal)[idx_test_sorted], aspect='auto')
    ax_main.set_xlabel("Hour")
    ax_main.set_ylabel("Example")
    ax_main.axhline(y=y_true.sum(), c='m')

    if ax is None:
        plt.show()
