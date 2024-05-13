import hashlib
import json
import os
import os.path as osp
import sys
from typing import Any, Dict, List, Tuple, Union

import torch
import yaml  # type: ignore
from icecream import ic
from torch import Tensor
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import FB15k_237, Planetoid, WordNet18RR
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


def convert_indices_to_english(
    tensor: Tensor,
    dictionary: Dict[int, str],
    is_entities: bool = True,
    show_only_first: bool = True,
) -> Union[str, List[str]]:
    """
    Converts indices in a tensor to English labels using a dictionary mapping.

    Args:
        tensor (Tensor): Tensor containing indices to convert.
        dictionary (Dict[int, str]): Dictionary mapping indices to labels.
        is_entities (bool): Flag to determine if the indices are entity IDs needing special conversion.
        show_only_first (bool): Flag to return only the first converted label.

    Returns:
        Union[str, List[str]]: A single label or a list of labels.
    """

    def map_to_string(index: int, dict: Dict[int, str]) -> str:
        return dict.get(index, "Not Found")

    if is_entities:
        if show_only_first:
            result = get_english_from_freebase_id(
                map_to_string(tensor[0], dictionary)
            )
        else:
            result = [
                get_english_from_freebase_id(map_to_string(index, dictionary))
                for index in tensor
            ]  # type: ignore
    else:
        if tensor.ndim == 1:  # Handle 1D tensor (e.g., [batch_size])
            if show_only_first:
                result = map_to_string(tensor[0], dictionary)
            else:
                result = [map_to_string(index, dictionary) for index in tensor]  # type: ignore
        else:  # Handle 2D tensor (e.g., [batch_size, num_candidates])
            if show_only_first:
                result = map_to_string(tensor[0][0], dictionary)
            else:
                result = [
                    [map_to_string(idx, dictionary) for idx in batch]
                    for batch in tensor
                ]  # type: ignore

    return result


def get_english_from_freebase_id(freebase_id: str) -> str:
    """
    Retrieves the English label for a given Freebase ID using Wikidata.

    Args:
        freebase_id (str): The Freebase ID to convert.

    Returns:
        str: The English label or an error message if not found.
    """
    try:
        entity_converter = EntityConverter("https://query.wikidata.org/sparql")
        res = entity_converter.get_wikidata_id(freebase_id)
        item = Client().get(res)
        return (
            item.label
            if item and item.label
            else f"No English found for this Freebase ID: {freebase_id}"
        )
    except AssertionError:
        return (
            f"This Freebase ID: {freebase_id} has no corresponding Wikidata ID"
        )


def load_dicts(
    data_path: str,
) -> Tuple[Dict[Any, Any] | None, Dict[Any, Any] | None]:
    """
    Loads entity and relation dictionaries from a specified path.

    Args:
        data_path (str): Path to the directory containing the dictionaries.

    Returns:
        Tuple[Dict[Any, Any] | None, Dict[Any, Any] | None]: Entity and relation dictionaries.
    """

    def load_dict_from_pt(file_path: str) -> Dict[Any, Any] | None:
        data = torch.load(file_path)
        return data if isinstance(data, dict) else None

    entity_dict = load_dict_from_pt(
        osp.join(
            osp.join(data_path, "processed"), "entity_dict.pt"
        )  # entity dict matches index: freebaseID
    )
    relation_dict = load_dict_from_pt(
        osp.join(osp.join(data_path, "processed"), "relation_dict.pt")
    )

    return entity_dict, relation_dict


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


def get_model_class(model_name: str) -> KGEModel:
    """
    Retrieves the class of the specified embedding model.

    Args:
        model_name (str): Name of the model to retrieve.

    Returns:
        KGEModel: The class of the specified model.
    """

    def get_embedding_class(model_name: str) -> KGEModel:
        if model_name == "RotatE":
            return CustomRotatE
        elif model_name == "TransE":
            return CustomTransE
        elif model_name == "DistMult":
            return CustomDistMult
        elif model_name == "ComplEx":
            return CustomComplEx
        else:
            raise ValueError(f"Model: {model_name} not supported")

    base_model_class = get_embedding_class(model_name)
    return lambda *args, **kwargs: base_model_class(*args, **kwargs)


def generate_unique_string(
    config: Dict[str, Any], first_x_characters: int = 8
) -> str:
    """
    Generates a unique string based on the hash of the configuration dictionary.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        first_x_characters (int): Number of characters to use from the hash.

    Returns:
        str: A unique string derived from the hash of the configuration.
    """
    # Ensure config is a dictionary
    if not isinstance(config, dict):
        config = dict(config)

    # Convert config to a sorted JSON string to ensure consistent ordering
    config_str = json.dumps(config, sort_keys=True)

    # Create a hash of the configuration
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[
        :first_x_characters
    ]  # Use only the first first_x_characters characters for brevity

    return config_hash


def save_model_config(model: KGEModel) -> str:
    """
    Saves the configuration of a model to a YAML file.

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


def save_trained_embedding_model_and_config(
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

    # Save the model state dictionary with additional attributes
    model_state_dict = model.state_dict()
    model_state_dict["num_nodes"] = model.train_data.num_nodes
    model_state_dict["num_relations"] = model.train_data.num_edge_types
    model_state_dict["hidden_channels"] = model.config["hidden_channels"]
    model_state_dict["dataset_name"] = model.config["dataset_name"]
    model_state_dict["embedding_model_name"] = model.config[
        "embedding_model_name"
    ]

    # Save the model as a .pth file
    torch.save(model_state_dict, model_weights_path)

    # Save the number of trained epochs and performance metrics in a single text file
    with open(performance_path, "w") as file:
        file.write(f"Trained Epochs: {epochs_trained}\n")
        file.write("Performance Metrics:\n")
        for metric, value in performance_metrics.items():
            file.write(f"{metric}: {value}\n")

    # Optionally return paths if needed
    return model_weights_path, performance_path
