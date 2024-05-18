import hashlib
import json
import random
from datetime import datetime
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import torch
import wandb
from torch_geometric.nn import KGEModel

from .embedding_models.ComplEx import CustomComplEx
from .embedding_models.DistMult import CustomDistMult
from .embedding_models.RotatE import CustomRotatE
from .embedding_models.TransE import CustomTransE


def set_seed(seed_value: int = 123) -> None:
    """Set the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_prefix(
    config: Dict[str, Any], run_timestamp: str, first_x_characters: int = 8
) -> str:
    """
    Generates a unique string based on config and the hash of config.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        run_timestamp (str): Timestamp of the run.
        first_x_characters (int): Number of characters to use from the hash.

    Returns:
        str: A prefix that describes the run.
    """
    if not isinstance(config, dict):
        config = dict(config)

    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[
        :first_x_characters
    ]

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


def run_sweep_or_main(
    run_sweep: bool,
    project_name: str,
    main: Callable,
    config: Optional[dict] = None,
) -> None:
    """
    Runs a hyperparameter sweep or the main training process based on the provided flag.

    Args:
        run_sweep (bool): Flag indicating whether to run a hyperparameter sweep.
        config (Optional[dict]): Configuration dictionary for the main training process.

    Returns:
        None
    """
    if run_sweep:
        run_timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        sweep_id = wandb.sweep(
            project=f"{project_name}_{run_timestamp}",
            sweep=config,
        )

        wandb.agent(sweep_id, function=main)
    else:
        main(config=config)
