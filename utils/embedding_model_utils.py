import hashlib
import json
import os
import os.path as osp
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch
import yaml  # type: ignore
from icecream import ic
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import FB15k_237
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WordNet18
from torch_geometric.datasets import WordNet18RR
from torch_geometric.nn import KGEModel

from .datasets.Planetoid import PlanetoidWithAuxiliaryNodes
from .datasets.YAGO3_10 import YAGO3_10
from .embedding_models.ComplEx import CustomComplEx
from .embedding_models.DistMult import CustomDistMult
from .embedding_models.RotatE import CustomRotatE
from .embedding_models.TransE import CustomTransE

# Add the path to import modules from the 'freebase' directory
sys.path.append(os.path.join(os.path.dirname(__file__), "freebase"))

from converter import EntityConverter
from wikidata.client import Client


def load_dataset(
    dataset_name: str, parent_dir: str, device: torch.device
) -> Tuple[Data, Data, Data, str]:
    """
    Loads the specified dataset and returns train, validation, and test splits along with the data path.

    Args:
        dataset_name (str): Name of the dataset to load.
        parent_dir (str): Parent directory where datasets are stored.
        device (torch.device): Device to which the data will be moved.

    Returns:
        Tuple[Data, Data, Data, str]: Train, validation, and test data splits and the path to the dataset.
    """
    data_path = os.path.join(parent_dir, "data", dataset_name)
    raw_dir = os.path.join(data_path, "raw")
    processed_dir = os.path.join(data_path, "processed")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Load specific datasets based on the name provided
    if dataset_name == "FB15k_237":
        train_data = FB15k_237(data_path, split="train")[0]
        val_data = FB15k_237(data_path, split="val")[0]
        test_data = FB15k_237(data_path, split="test")[0]
    elif dataset_name == "WordNet18RR":
        dataset = WordNet18RR(root=data_path)
        data = dataset[
            0
        ]  # Assuming the dataset is loaded and processed correctly
        train_data = data.clone()
        val_data = data.clone()
        test_data = data.clone()

        # Apply masks to get the respective splits
        train_data.edge_index = data.edge_index[:, data.train_mask]
        train_data.edge_type = data.edge_type[data.train_mask]
        val_data.edge_index = data.edge_index[:, data.val_mask]
        val_data.edge_type = data.edge_type[data.val_mask]
        test_data.edge_index = data.edge_index[:, data.test_mask]
        test_data.edge_type = data.edge_type[data.test_mask]
    elif dataset_name == "WordNet18":
        dataset = WordNet18(root=data_path)
        data = dataset[
            0
        ]  # Assuming the dataset is loaded and processed correctly
        train_data = data.clone()
        val_data = data.clone()
        test_data = data.clone()

        # Apply masks to get the respective splits
        train_data.edge_index = data.edge_index[:, data.train_mask]
        train_data.edge_type = data.edge_type[data.train_mask]
        val_data.edge_index = data.edge_index[:, data.val_mask]
        val_data.edge_type = data.edge_type[data.val_mask]
        test_data.edge_index = data.edge_index[:, data.test_mask]
        test_data.edge_type = data.edge_type[data.test_mask]
    elif dataset_name == "YAGO3_10":
        train_data = YAGO3_10(data_path, split="train")[0]
        val_data = YAGO3_10(data_path, split="val")[0]
        test_data = YAGO3_10(data_path, split="test")[0]
    elif dataset_name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = PlanetoidWithAuxiliaryNodes(
            root=data_path, name=dataset_name
        )
        dataset.load(dataset.processed_paths[0])
        data = dataset[0]

        aux_dict = dataset.aux_node_to_label
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

        train_features = data.x[train_mask]
        val_features = data.x[val_mask]
        test_features = data.x[test_mask]

        train_labels = data.y[train_mask]
        val_labels = data.y[val_mask]
        test_labels = data.y[test_mask]

        train_edge_index = data.edge_index[:, train_mask]
        train_edge_type = torch.ones(
            train_edge_index.size(1), dtype=torch.int64
        )
        val_edge_index = data.edge_index[:, val_mask]
        val_edge_type = torch.ones(val_edge_index.size(1), dtype=torch.int64)
        test_edge_index = data.edge_index[:, test_mask]
        test_edge_type = torch.ones(test_edge_index.size(1), dtype=torch.int64)

        train_data = Data(
            x=train_features,
            edge_index=train_edge_index,
            edge_type=train_edge_type,
            y=train_labels,
            aux_dict=aux_dict,
            num_edge_types=2,
        )
        val_data = Data(
            x=val_features,
            edge_index=val_edge_index,
            edge_type=val_edge_type,
            y=val_labels,
            aux_dict=aux_dict,
            num_edge_types=2,
        )
        test_data = Data(
            x=test_features,
            edge_index=test_edge_index,
            edge_type=test_edge_type,
            y=test_labels,
            aux_dict=aux_dict,
            num_edge_types=2,
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return (
        train_data.to(device),
        val_data.to(device),
        test_data.to(device),
        data_path,
    )


def fetch_and_prepare_batch(
    data_loader: DataLoader, device: torch.device
) -> Union[Tuple[Tensor, ...], Dict[str, Tensor]]:
    """
    Fetches the first batch from the specified DataLoader and moves it to the given device.

    Args:
        data_loader (DataLoader): The DataLoader from which to fetch the batch.
        device (torch.device): The device to which the batch should be moved.

    Returns:
        Union[Tuple[Tensor, ...], Dict[str, Tensor]]: The first batch from the DataLoader, with all components moved to the specified device.
    """
    for batch in data_loader:
        # Assuming batch contains head_index, rel_type, tail_index, and possibly other data
        if isinstance(batch, (list, tuple)):
            # Move all elements in the batch to the specified device
            batch = tuple(item.to(device) for item in batch)
        elif isinstance(batch, dict):
            # Move all tensor values in the dictionary to the specified device
            batch = {
                key: (
                    value.to(device)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in batch.items()
            }
        else:
            raise NotImplementedError(
                "Batch for this purpose needs to be either a tuple or dict"
            )
        break  # Only need one batch for this purpose
    return batch


def save_embedding_model_config(model: KGEModel) -> str:
    """
    Saves the configuration of a embedding model to a YAML file.

    Args:
        model (KGEModel): The model whose configuration is to be saved.

    Returns:
        str: Path to the saved configuration file.
    """
    config = model.config
    save_path = model.save_path
    # Convert wandb.config to a dictionary if it's not already one
    if not isinstance(config, dict):
        config = dict(config)

    # Define where we are saving the dumped config
    config_path = osp.join(
        save_path, f"{model.config['prefix']}_embedding_model_config.yaml"
    )

    # Save the redacted config file
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    return config_path


def save_trained_embedding_weights_and_performance(
    model: KGEModel, epochs_trained: int, performance_metrics: Dict[str, float]
) -> Tuple[str, str]:
    """
    Saves the trained model's weights and performance metrics to disk.

    Args:
        model (KGEModel): The trained model instance.
        epochs_trained (int): The number of epochs the model was trained for.
        performance_metrics (Dict[str, float]): A dictionary containing performance metrics.

    Returns:
        Tuple[str, str]: Paths to the saved model weights and performance metrics files.
    """
    # Define the file paths for model weights and combined performance and epochs file
    model_weights_path = osp.join(
        model.save_path,
        f"{model.config['prefix']}_embedding_model_weights.pth",
    )
    performance_path = osp.join(
        model.save_path,
        f"{model.config['prefix']}_embedding_model_performance.txt",
    )

    # Save the model as a .pth file
    torch.save(model.state_dict(), model_weights_path)

    # Save the number of trained epochs and performance metrics in a single text file
    with open(performance_path, "w") as file:
        file.write(f"Trained Epochs: {epochs_trained}\n")
        file.write("Performance Metrics:\n")
        for metric, value in performance_metrics.items():
            file.write(f"{metric}: {value}\n")

    # Optionally return paths if needed
    return model_weights_path, performance_path
