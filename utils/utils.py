import os
import os.path as osp
import sys

import torch
import torch_geometric.transforms as T
from icecream import ic
from torch_geometric.data import Data
from torch_geometric.datasets import FB15k_237, PCQM4Mv2, WordNet18RR
from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE

# Add the path to import modules from the 'freebase' directory
sys.path.append(os.path.join(os.path.dirname(__file__), "freebase"))

from converter import EntityConverter
from wikidata.client import Client


def load_dataset(dataset_name, parent_dir, device):
    data_path = osp.join(parent_dir, "data", dataset_name)
    if dataset_name == "FB15k_237":
        return (
            FB15k_237(data_path, split="train")[0].to(device),
            FB15k_237(data_path, split="val")[0].to(device),
            FB15k_237(data_path, split="test")[0].to(device),
            data_path,
        )
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
        return (
            train_data.to(device),
            val_data.to(device),
            test_data.to(device),
            data_path,
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def get_model_class(model_name):
    if model_name == "RotatE":
        return RotatE
    elif model_name == "TransE":
        return TransE
    elif model_name == "ComplEx":
        return ComplEx
    elif model_name == "DistMult":
        return DistMult
    else:
        raise ValueError(f"Model: {model_name} not supported")


def convert_indices_to_english(
    tensor, dictionary, is_entities=True, show_only_first=True
):
    def map_to_string(index, dict):
        return dict.get(index.item(), "Not Found")

    if is_entities:
        if show_only_first:
            result = get_english_from_freebase_id(
                map_to_string(tensor[0], dictionary)
            )
        else:
            result = [
                get_english_from_freebase_id(map_to_string(index, dictionary))
                for index in tensor
            ]
    else:
        if tensor.ndim == 1:  # Handle 1D tensor (e.g., [batch_size])
            if show_only_first:
                result = map_to_string(tensor[0], dictionary)
            else:
                result = [map_to_string(index, dictionary) for index in tensor]
        else:  # Handle 2D tensor (e.g., [batch_size, num_candidates])
            if show_only_first:
                result = map_to_string(tensor[0][0], dictionary)
            else:
                result = [
                    [map_to_string(idx, dictionary) for idx in batch]
                    for batch in tensor
                ]

    return result


def get_english_from_freebase_id(freebase_id):
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


def load_dicts(data_path):
    def load_dict_from_pt(file_path):
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
