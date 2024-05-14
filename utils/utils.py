import hashlib
import json
import os
import os.path as osp
import random
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
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


def set_seed(seed_value=123):
    """Set the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_prefix(
    config: Dict[str, Any], run_timestamp: str, first_x_characters: int = 8
) -> str:
    """
    Generates a unique string based on config and the hash of config.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        first_x_characters (int): Number of characters to use from the hash.

    Returns:
        str: A prefix that describes the run
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

    return f'{config["dataset_name"]}_{config["embedding_model_name"]}_{config["task"]}_{run_timestamp}_{config_hash}'


def get_embedding_model_class(model_name: str) -> KGEModel:
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
