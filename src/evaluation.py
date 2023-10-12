import torch
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def fgsm_attack(
    input_: torch.Tensor, epsilon: torch.Tensor, data_grad: torch.Tensor
) -> torch.Tensor:
    sign_data_grad = data_grad.sign()
    perturbed_input = input_ + epsilon * sign_data_grad
    return perturbed_input


def test_attack(
    model: torch.nn.Module,
    data_: torch.Tensor,
    labels_: torch.Tensor,
    epsilon: torch.Tensor,
    device: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    criterion = torch.nn.CrossEntropyLoss().to(device)
    correct = 0

    data_ = data_.to(device)
    labels_ = labels_.to(device)
    data_.requires_grad = True

    output = model(data_)
    loss = criterion(output, labels_)
    model.zero_grad()
    loss.backward()
    data_grad = data_.grad.data
    with torch.no_grad():
        perturbed_data = fgsm_attack(data_, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.argmax(dim=1)
        correct += (final_pred == labels_).sum().item()

    return (
        correct / float(len(data_)),
        perturbed_data.cpu().detach().numpy(),
        final_pred.cpu().detach().numpy(),
    )


def eval_model(
    model_name: str,
    model: torch.nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str,
) -> float:
    with torch.no_grad():
        model.eval()
        outputs = model(X_test.to(device))
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        accuracy = balanced_accuracy_score(y_test.cpu().numpy(), predictions)
        print(f"{model_name} test accuracy = {accuracy}")
    return accuracy


def get_saliencies(
    model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, device: str
) -> tuple[np.ndarray, np.ndarray]:
    for p in model.parameters():
        p.requires_grad_(False)

    X = X.to(device)
    y = y.to(device)
    X.requires_grad_(True)

    preds = model(X)
    preds[:, 0].sum().backward()
    saliencies_0 = X.grad.cpu().numpy()
    X.grad = torch.zeros_like(X.grad)

    model(X)[:, 1].sum().backward()
    saliencies_1 = X.grad.cpu().numpy()
    X.grad = torch.zeros_like(X.grad)
    X.requires_grad_(False)

    return saliencies_0, saliencies_1


def vizualize_saliency(
    df: torch.Tensor,
    labels: torch.Tensor,
    preds: np.ndarray,
    index_: int,
    saliencies_0: np.ndarray,
    saliencies_1: np.ndarray,
) -> None:
    x = df[index_, :, :].squeeze().cpu().detach().numpy()
    sal_0 = saliencies_0[index_, :, :]
    sal_1 = saliencies_1[index_, :, :]

    y_0 = (sal_0 - sal_0.min()) / (sal_0.max() - sal_0.min())
    y_1 = (sal_1 - sal_1.min()) / (sal_1.max() - sal_1.min())

    fig, ax = plt.subplots(ncols=2, figsize=(15, 4))

    fig.suptitle(f"True label is {labels[index_]}. Predicted label is {preds[index_]}")
    ax[0].plot(np.arange(len(x)), x)
    scatter_0 = ax[0].scatter(
        np.arange(len(x)), x, c=y_0, cmap="viridis", marker="o", s=100
    )
    plt.colorbar(mappable=scatter_0, ax=ax[0], label="Saliency")

    ax[0].set_title(f"Saliencies wrt class 0.")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Power Demand")

    ax[1].plot(np.arange(len(x)), x)
    scatter_1 = ax[1].scatter(
        np.arange(len(x)), x, c=y_1, cmap="viridis", marker="o", s=100
    )
    plt.colorbar(mappable=scatter_1, ax=ax[1], label="Saliency")

    ax[1].set_title(f"Saliencies wrt class 1.")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Power Demand")

    plt.tight_layout()
    plt.show()
