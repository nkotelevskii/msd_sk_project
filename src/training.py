from tqdm.auto import tqdm

from src.models import LSTM
import torch.nn as nn
import torch
from torch import optim
from sklearn.metrics import balanced_accuracy_score
import copy


def trainer(
    model: nn.Module,
    n_epochs: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: str,
    lr: float = 1e-4,
    tolerance: int = 10,
    freq_print: int = 25
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)
    X_train = X_train.float().to(device)
    y_train = y_train.to(device)
    X_val = X_val.float().to(device)
    y_val = y_val.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_accuracy = 0
    counter = 0
    best_model = copy.deepcopy(model)
    
    with tqdm(range(n_epochs)) as pbar:

        for epoch in pbar:
            pbar.set_description(f"Epoch: {epoch}")
            model.train()
            optimizer.zero_grad()

            outputs = model(X_train)
            loss = criterion(outputs, y_train.squeeze())
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                outputs = model(X_val)
                _, predictions = torch.max(outputs.data, 1)
                predictions = predictions.cpu().detach().numpy()

                train_acc = balanced_accuracy_score(
                    y_train.cpu().numpy(),
                    torch.argmax(model(X_train), dim=1).cpu().detach().numpy(),
                )
                test_acc = balanced_accuracy_score(y_val.cpu().numpy(), predictions)
            pbar.set_postfix({"loss": loss.item(), "train_acc": train_acc, "val_acc": test_acc})
            """if (epoch + 1) % freq_print == 0:
                print(
                    (
                        f"Epoch = {epoch}, \t"
                        f"Loss = {loss}, \t"
                        f"Training_acc = {train_acc}, \t"
                        f"Validation_acc = {test_acc}"
                    )
                )"""
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_model = copy.deepcopy(model)
                counter = 0
            else:
                counter += 1
                if counter == tolerance:
                    break

    print(f"Best validation accuracy is: {best_accuracy}")
    return best_model
