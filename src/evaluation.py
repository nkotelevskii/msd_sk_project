import torch
from sklearn.metrics import balanced_accuracy_score


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
