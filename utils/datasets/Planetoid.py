import json
import os.path as osp
import typing
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from icecream import ic
from torch_geometric.data import Data
from torch_geometric.datasets.planetoid import Planetoid
from torch_geometric.io import fs, read_planetoid_data


class PlanetoidWithAuxiliaryNodes(Planetoid):
    """
    A class to handle Planetoid datasets with additional auxiliary nodes for each label.
    This is useful for graph neural network models that require additional nodes for specific tasks.

    Attributes:
        label_to_aux_node (Dict[int, int]): Mapping from labels to auxiliary node indices.
        aux_node_to_label (Dict[int, int]): Mapping from auxiliary node indices to labels.
    """

    def __init__(
        self,
        root: str,
        name: str,
        split: str = "public",
        num_train_per_class: int = 20,
        num_val: int = 500,
        num_test: int = 1000,
        transform: Optional[typing.Callable] = None,
        pre_transform: Optional[typing.Callable] = None,
        force_reload: bool = False,
    ):
        """
        Initializes the PlanetoidWithAuxiliaryNodes dataset.

        Args:
            root (str): Root directory where the dataset should be stored.
            name (str): The name of the dataset (e.g., 'Cora', 'CiteSeer').
            split (str): The type of dataset split ('public', 'random', or 'geom-gcn').
            num_train_per_class (int): Number of training examples per class (only used in 'public' and 'random' splits).
            num_val (int): Number of validation examples.
            num_test (int): Number of test examples.
            transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version.
            pre_transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version. The function is applied before saving the data to disk.
            force_reload (bool): If True, the dataset will be re-downloaded and processed.
        """
        super().__init__(
            root,
            name,
            split,
            num_train_per_class,
            num_val,
            num_test,
            transform,
            pre_transform,
            force_reload,
        )
        self.label_to_aux_node: Dict[int, int] = {}
        self.aux_node_to_label: Dict[int, int] = {}

    @property
    def raw_dir(self) -> str:
        """Returns the directory where raw files are stored."""
        if self.split == "geom-gcn":
            return osp.join(self.root, self.name, "geom-gcn", "raw")
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        """Returns the directory where processed files are stored."""
        if self.split == "geom-gcn":
            return osp.join(self.root, self.name, "geom-gcn", "processed")
        return osp.join(self.root, "processed")

    def process(self) -> None:
        """
        Processes the raw data files and saves them into a processed file.
        """
        data = read_planetoid_data(self.raw_dir, self.name)
        if self.split == "geom-gcn":
            train_masks, val_masks, test_masks = [], [], []
            for i in range(10):
                name = f"{self.name.lower()}_split_0.6_0.2_{i}.npz"
                splits = np.load(osp.join(self.raw_dir, name))
                train_masks.append(torch.from_numpy(splits["train_mask"]))
                val_masks.append(torch.from_numpy(splits["val_mask"]))
                test_masks.append(torch.from_numpy(splits["test_mask"]))
            data.train_mask = torch.stack(train_masks, dim=1)
            data.val_mask = torch.stack(val_masks, dim=1)
            data.test_mask = torch.stack(test_masks, dim=1)

        self.label_to_aux_node, self.aux_node_to_label = (
            self.create_label_mappings(data)
        )
        data = self.create_train_val_test_data(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        path = self.processed_paths[0]  # Assuming a single processed file
        aux_node_to_label_path = osp.join(
            osp.dirname(path), "aux_node_to_label.json"
        )
        with open(
            aux_node_to_label_path, "w"
        ) as f:  # Save the aux_node mapping as a JSON file
            json.dump(self.aux_node_to_label, f)
        self.save([data], path)

    @staticmethod
    def load_aux_node_to_label(processed_dir: str) -> Dict[int, int]:
        """
        Loads the auxiliary node to label mapping from a JSON file.

        Args:
            processed_dir (str): The directory where the processed files are stored.

        Returns:
            Dict[int, int]: A dictionary mapping auxiliary node indices to labels.
        """
        aux_node_to_label_path = osp.join(
            processed_dir, "aux_node_to_label.json"
        )
        with open(aux_node_to_label_path) as f:
            aux_node_to_label = json.load(f)
        return aux_node_to_label

    def load(self, path: str, data_cls: Type[Data] = Data) -> None:
        """
        Loads the dataset from the specified file path and attaches the auxiliary node to label mapping.

        Args:
            path (str): The path to the file from which to load the dataset.
            data_cls (Type[Data]): The class for data handling. Defaults to torch_geometric.data.Data.

        Raises:
            FileNotFoundError: If the auxiliary node to label JSON file is not found.
        """
        out = fs.torch_load(path)
        if not isinstance(out, tuple):
            raise TypeError("Expected 'out' to be a tuple.")

        if len(out) not in [2, 3]:
            raise ValueError("Expected 'out' to have a length of 2 or 3.")
        if len(out) == 2:  # Backward compatibility.
            data, self.slices = out
        else:
            data, self.slices, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        # Load aux_node_to_label if present
        aux_node_to_label_path = osp.join(
            osp.dirname(path), "aux_node_to_label.json"
        )
        if osp.exists(aux_node_to_label_path):
            with open(aux_node_to_label_path) as f:
                self.aux_node_to_label = json.load(f)
        else:
            raise FileNotFoundError(
                "aux_node_to_label JSON file not found in the same directory as the processed data."
            )

    def create_label_mappings(
        self, data: Data
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Creates mappings from labels to auxiliary nodes and vice versa.

        Args:
            data (Data): The data object containing node features and labels.

        Returns:
            Tuple[Dict[int, int], Dict[int, int]]: A tuple containing two dictionaries for label to auxiliary node mapping and auxiliary node to label mapping.
        """
        label_to_aux_node = {}
        aux_node_to_label = {}
        num_nodes = data.x.size(0)
        for label in data.y.unique():
            label_to_aux_node[int(label)] = num_nodes
            aux_node_to_label[num_nodes] = int(label)
            num_nodes += 1
        return label_to_aux_node, aux_node_to_label

    def create_train_val_test_data(self, data: Data) -> Data:
        """
        Modifies the dataset to include auxiliary nodes and updates masks for training, validation, and testing.

        Args:
            data (Data): The original data object.

        Returns:
            Data: The modified data object with additional auxiliary nodes and updated masks.
        """
        # Generate auxiliary edges based on the masks
        train_aux_edges, train_aux_count = self.create_aux_node_edges(
            data, data.train_mask
        )
        val_aux_edges, val_aux_count = self.create_aux_node_edges(
            data, data.val_mask
        )
        test_aux_edges, test_aux_count = self.create_aux_node_edges(
            data, data.test_mask
        )

        original_num_nodes = data.x.size(0)
        original_num_edges = data.edge_index.size(1)

        new_mask_size = original_num_edges + original_num_nodes

        # Initialize new_edge_index with original edges and space for potential new edges
        new_edge_index = torch.full((2, new_mask_size), -1, dtype=torch.long)
        new_edge_index[:, :original_num_edges] = (
            data.edge_index
        )  # Fill in original edges

        # Iterate over each node to potentially add new edges
        for node_idx in range(original_num_nodes):
            if (
                data.train_mask[node_idx]
                or data.val_mask[node_idx]
                or data.test_mask[node_idx]
            ):
                aux_node_idx = self.label_to_aux_node[int(data.y[node_idx])]
                new_edge_index[0, original_num_edges + node_idx] = node_idx
                new_edge_index[1, original_num_edges + node_idx] = aux_node_idx

        # Initialize new masks
        new_x = torch.zeros((new_mask_size, data.x.size(1)))
        new_y = torch.zeros(new_mask_size, dtype=torch.long)

        new_train_mask = torch.zeros(new_mask_size, dtype=torch.bool)
        new_val_mask = torch.zeros(new_mask_size, dtype=torch.bool)
        new_test_mask = torch.zeros(new_mask_size, dtype=torch.bool)

        # Copy original features and labels and update masks
        new_x[original_num_edges:] = data.x
        new_y[original_num_edges:] = data.y
        new_train_mask[:original_num_edges] = True
        new_train_mask[original_num_edges:] = data.train_mask
        new_val_mask[original_num_edges:] = data.val_mask
        new_test_mask[original_num_edges:] = data.test_mask

        # Update the data object
        data.x = new_x
        data.y = new_y
        data.edge_index = new_edge_index
        data.train_mask = new_train_mask
        data.val_mask = new_val_mask
        data.test_mask = new_test_mask
        return data

    def create_aux_node_edges(
        self, data: Data, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Creates edges between nodes and their corresponding auxiliary nodes based on the given mask.

        Args:
            data (Data): The data object containing node features and labels.
            mask (torch.Tensor): A mask indicating which nodes to connect to auxiliary nodes.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the tensor of new edges and the count of these edges.
        """
        aux_node_edges = []
        for node_idx in mask.nonzero().view(-1):
            if node_idx < data.y.size(
                0
            ):  # Ensure the node index is valid for labels
                aux_node_idx = self.label_to_aux_node[int(data.y[node_idx])]
                aux_node_edges.append([node_idx, aux_node_idx])

        aux_node_edges_tensor = torch.tensor(
            aux_node_edges, dtype=torch.long
        ).t()
        return aux_node_edges_tensor, len(aux_node_edges)
