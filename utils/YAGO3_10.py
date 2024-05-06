import os
import os.path as osp
import shutil
import urllib.request
import zipfile
from collections import defaultdict

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url


class YAGO3_10(InMemoryDataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.entities = {}
        self.relations = {}
        # for some reason, you need to set entities and relations prior to super since super calls download and process
        super().__init__(root, transform, pre_transform, pre_filter)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid 'split' argument (got {split})")
        split_index = {"train": 0, "val": 1, "test": 2}[split]
        self.data, self.slices = torch.load(self.processed_paths[split_index])

    def __getitem__(self, idx):
        return torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]

    @property
    def processed_file_names(self):
        return [
            "train_data.pt",
            "val_data.pt",
            "test_data.pt",
            "entities.pt",
            "relations.pt",
        ]

    @property
    def raw_paths(self):
        return [osp.join(self.raw_dir, fname) for fname in self.raw_file_names]

    def download(self):
        url = "https://ampligraph.s3-eu-west-1.amazonaws.com/datasets/YAGO3-10.zip"
        zip_path = download_url(url, self.raw_dir)

        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)

        # Move the extracted files to the raw_dir and remove the zip file and extracted directory
        extracted_dir = osp.join(self.raw_dir, "YAGO3-10")
        for fname in self.raw_file_names:
            src = osp.join(extracted_dir, fname)
            dst = osp.join(self.raw_dir, fname)
            shutil.move(src, dst)

        os.remove(zip_path)
        shutil.rmtree(extracted_dir)

    def process(self):
        entities = {}
        relations = {}

        # First pass: collect unique nodes and relations
        for split_file in self.raw_paths:
            with open(split_file) as f:
                lines = [x.strip().split("\t") for x in f.readlines()]

            src_list, rel_list, dst_list = zip(*lines)
            unique_nodes = set(src_list + dst_list)
            unique_relations = set(rel_list)

            entities.update(
                {
                    node: len(entities) + idx
                    for idx, node in enumerate(unique_nodes)
                }
            )
            relations.update(
                {rel: idx for idx, rel in enumerate(unique_relations)}
            )

        # Second pass: construct the edge index tensor
        for idx, split_file in enumerate(self.raw_paths):
            with open(split_file) as f:
                lines = [x.strip().split("\t") for x in f.readlines()]

            src_list, rel_list, dst_list = zip(*lines)

            src_idx = torch.tensor(
                [entities[src] for src in src_list], dtype=torch.long
            )
            dst_idx = torch.tensor(
                [entities[dst] for dst in dst_list], dtype=torch.long
            )
            rel_idx = torch.tensor(
                [relations[rel] for rel in rel_list], dtype=torch.long
            )

            edge_index = torch.stack([src_idx, dst_idx], dim=0)
            edge_type = rel_idx

            data = Data(edge_index=edge_index, edge_type=edge_type)
            split_data_file = osp.join(
                self.processed_dir, self.processed_file_names[idx]
            )
            torch.save(data, split_data_file)

        torch.save(entities, osp.join(self.processed_dir, "entities.pt"))
        torch.save(relations, osp.join(self.processed_dir, "relations.pt"))
