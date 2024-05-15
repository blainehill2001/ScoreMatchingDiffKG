import hashlib
import json
import os
import os.path as osp
import sys
from collections import OrderedDict
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
from .utils import get_embedding_model_class

# Add the path to import modules from the 'freebase' directory
sys.path.append(osp.join(osp.dirname(__file__), "freebase"))

from converter import EntityConverter
from wikidata.client import Client


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def extract_info_from_string(embedding_model_dir_string):

    # Extract the prefix from the directory string
    prefix = osp.basename(osp.normpath(embedding_model_dir_string))

    # Load config.yaml file
    config_path = osp.join(embedding_model_dir_string, f"{prefix}_config.yaml")
    config = load_config(config_path)

    # Extract dataset name, embedding model name, and task
    dataset_name = config.get("dataset_name", None)
    embedding_model_name = config.get("embedding_model_name", None)
    task = config.get("task", None)
    aux_dict = config.get("aux_dict", None)

    return dataset_name, embedding_model_name, task, aux_dict


def initialize_trained_embedding_model(embedding_model_dir_string, device):
    # Extract the prefix from the directory string
    prefix = osp.basename(osp.normpath(embedding_model_dir_string))

    # Construct file names using the prefix
    config_file = f"{prefix}_config.yaml"
    weights_file = f"{prefix}_weights.pth"

    # Construct full file paths
    config_path = osp.join(embedding_model_dir_string, config_file)
    weights_path = osp.join(embedding_model_dir_string, weights_file)

    # Load configuration from YAML file
    config = load_config(config_path)

    # Extract necessary information from the config
    num_nodes = config.get("num_nodes")
    num_relations = config.get("num_relations")
    hidden_channels = config.get("hidden_channels")
    embedding_model_name = config.get("embedding_model_name")
    dataset_name = config.get("dataset_name")

    if any(
        x is None
        for x in [
            num_nodes,
            num_relations,
            hidden_channels,
            embedding_model_name,
            dataset_name,
        ]
    ):
        raise ValueError(
            "embedding model num_nodes, num_relations, hidden_channels, embedding_model_name, or dataset_name not found in the config file"
        )

    # Get the model class
    embedding_model_class = get_embedding_model_class(embedding_model_name)

    # Initialize the actual embedding model with config parameters
    model_init_params = {
        "num_nodes": num_nodes,
        "num_relations": num_relations,
        "hidden_channels": hidden_channels,
    }

    # Add optional parameters if they exist in the config
    if "aud_dict" in config:
        model_init_params["aud_dict"] = config["aud_dict"]
    if "head_node_feature_dim" in config:
        model_init_params["head_node_feature_dim"] = config[
            "head_node_feature_dim"
        ]

    embedding_model = embedding_model_class(**model_init_params)

    # Load the model state dictionary
    loaded_weights = torch.load(weights_path, map_location=device)

    # the weights file has the same information prefixed by both .module and .original_model due to DataParallel
    new_state_dict = OrderedDict()
    for k, v in loaded_weights.items():
        if k.startswith("module."):
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        # discard the "original_model." prefixed information since it is the same as module.
    if len(new_state_dict) == 0:
        raise ValueError(
            f"Weights stored in {weights_path} do not have any keys"
        )
    loaded_weights = new_state_dict

    # Load the state dictionary into the model
    embedding_model.load_state_dict(loaded_weights)
    embedding_model.to(device)

    return embedding_model, embedding_model_name, dataset_name


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


"""
# Below is an example of how you might use convert_indices_to_english and get_english_from_freebase_id
if view_english and self.dataset_name == "FB15k_237":
    # for getting english from freebase ID for FB15k_237 data
    # show_only_first=True means that only the first entry in the batch will be processed and returned in english
    ic(
        convert_indices_to_english(
            indices10, self.relation_dict, is_entities=False
        )
    )
    ic(
        convert_indices_to_english(
            true_r, self.relation_dict, is_entities=False
        )
    )
    ic(convert_indices_to_english(h, self.entity_dict))
    ic(convert_indices_to_english(t, self.entity_dict))
"""


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


def save_score_matching_model_config(model) -> str:
    """
    Saves the configuration of a score matching model to a YAML file.

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
        save_path, f"{model.config['prefix']}_score_matching_model_config.yaml"
    )

    # Save the redacted config file
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    return config_path


def save_trained_score_matching_weights_and_performance(
    model, epochs_trained: int, performance_metrics: Dict[str, float]
) -> Tuple[str, str]:
    """
    Saves the trained score matching model's weights and performance metrics to disk.

    Args:
        model (): The trained model instance.
        epochs_trained (int): The number of epochs the model was trained for.
        performance_metrics (Dict[str, float]): A dictionary containing performance metrics.

    Returns:
        Tuple[str, str]: Paths to the saved model weights and performance metrics files.
    """
    # Define the file paths for model weights and combined performance and epochs file
    model_weights_path = osp.join(
        model.save_path,
        f"{model.config['prefix']}_score_matching_model_weights.pth",
    )
    performance_path = osp.join(
        model.save_path,
        f"{model.config['prefix']}_score_matching_model_performance.txt",
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
