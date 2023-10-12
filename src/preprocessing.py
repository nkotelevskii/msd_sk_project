import numpy as np
import pandas as pd
import torch


def parse_data(
    data_path: str = "./data/ItalyPowerDemand_TRAIN.ts",
) -> tuple[torch.Tensor, torch.Tensor]:
    data_dict = {}
    current_key = None
    current_values = []

    with open(data_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("@"):
                if current_key is not None:
                    if len(current_values) == 1:
                        data_dict[current_key] = current_values[0]
                    else:
                        data_dict[current_key] = current_values
                    current_values = []
                current_key = line
            else:
                current_values.append(line)

    if current_key is not None:
        if len(current_values) == 1:
            data_dict[current_key] = current_values[0]
        else:
            data_dict[current_key] = current_values

    split_data = [item.split(":") for item in data_dict["@data"]]
    pdfeatures0 = [list(map(float, item[0].split(","))) for item in split_data]
    features0 = np.array([list(map(float, item[0].split(","))) for item in split_data])
    labels = [1 if int(item[-1]) == 1 else 0 for item in split_data]
    pddf = pd.DataFrame({"F1": pdfeatures0, "Labels": labels})
    df = np.array(features0)
    labels = np.array(labels, dtype=int)

    df_features = torch.tensor(df).float().unsqueeze(1)
    df_labels = torch.tensor(labels).long()
    df_features_t = torch.transpose(df_features, 1, 2)

    return df_features_t, df_labels


def split_train_test(
    X: torch.Tensor, y: torch.Tensor, objects_for_train: int = 34
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
    df_train_features, df_train_labels, df_test_features, df_test_labels = (
        X[:objects_for_train, :, :],
        y[:objects_for_train],
        X[objects_for_train:, :, :],
        y[objects_for_train:],
    )
    return df_train_features, df_train_labels, df_test_features, df_test_labels
