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
from .utils import get_embedding_model_class

# Add the path to import modules from the 'freebase' directory
sys.path.append(osp.join(osp.dirname(__file__), "freebase"))

from converter import EntityConverter
from wikidata.client import Client


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def extract_info_from_string(string):
    # Extract directory path
    dir_path = osp.dirname(string)

    # Load config.yaml file
    config_path = osp.join(dir_path, "config.yaml")
    config = load_config(config_path)

    # Extract dataset name, embedding model name, and task
    dataset_name = config.get("dataset_name", "")
    embedding_model_name = config.get("embedding_model_name", "")
    task = config.get("task", "")

    return dataset_name, embedding_model_name, task


def initialize_trained_embedding_model(
    embedding_model_dir,
):
    # Extract the prefix from the directory string
    prefix = osp.basename(osp.normpath(dir_string))

    # Construct file names using the prefix
    performance_file = f"{prefix}_performance.txt"
    config_file = f"{prefix}_config.yaml"
    weights_file = f"{prefix}_weights.pth"

    # Construct full file paths
    performance_path = osp.join(dir_string, performance_file)
    config_path = osp.join(dir_string, config_file)
    weights_path = osp.join(dir_string, weights_file)

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
    complete_state_dict = torch.load(state_dict_path, map_location=device)

    # Load the state dictionary into the model
    embedding_model.load_state_dict(model_state_dict)
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
