import math
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from icecream import ic
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class ScoreModel(nn.Module):
    def __init__(
        self,
        embedding_model,
        node_emb_dim,
        rel_emb_dim,
        score_model_hidden_dim=512,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        # Get entity and relation embeddings from embedding model
        with torch.no_grad():
            entity_embeddings = embedding_model.node_emb.weight.detach()
            relation_embeddings = embedding_model.rel_emb.weight.detach()
            feature_transform = (
                embedding_model.feature_transform_re.weight.detach()
            )

            # handle both real and imaginary node and relation embeddings
            if hasattr(embedding_model, "node_emb_im") and hasattr(
                embedding_model, "feature_transform_im"
            ):  # occurs with RotatE, ComplEx
                entity_embeddings_im = (
                    embedding_model.node_emb_im.weight.detach()
                )
                feature_transform_im = (
                    embedding_model.feature_transform_im.weight.detach()
                )
                entity_embeddings = torch.cat(
                    [entity_embeddings, entity_embeddings_im], dim=-1
                )
                feature_transform = torch.cat(
                    [feature_transform, feature_transform_im], dim=-1
                )

            if hasattr(embedding_model, "rel_emb_im"):  # occurs with ComplEx
                relation_embeddings_im = (
                    embedding_model.rel_emb_im.weight.detach()
                )
                relation_embeddings = torch.cat(
                    [relation_embeddings, relation_embeddings_im], dim=-1
                )
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.feature_transform = feature_transform
        self.node_emb_dim = (self.entity_embeddings.shape[-1],)
        self.rel_emb_dim = (self.relation_embeddings.shape[-1],)
        self.score_net = nn.Sequential(
            nn.Linear(
                self.entity_embeddings.shape[-1]
                + self.relation_embeddings.shape[-1],
                score_model_hidden_dim,
                dtype=torch.float,
            ),
            nn.ReLU(),
            nn.Linear(
                score_model_hidden_dim,
                score_model_hidden_dim,
                dtype=torch.float,
            ),
            nn.ReLU(),
            nn.Linear(score_model_hidden_dim, 1, dtype=torch.float),
        )

    def forward(self, h_emb, r_emb, t_emb, timestep=None):
        # Implement your desired distance measure here (e.g., L2 distance)
        distance = torch.linalg.norm(
            h_emb + r_emb - t_emb, dim=-1
        )  # semantic meaning only (no graph structural information was used to calculate distance)

        # Gradually increase the weight of the distance term during SDE steps
        weight = 0.0  # No weight increase if timestep is None
        if timestep is not None:

            def sigmoid(x):
                return 1 / (1 + math.exp(-x))

            weight = sigmoid(timestep / (self.config["num_timesteps"] - 1))
        score = weight * distance
        return score

    def loss(
        self,
        h: Tensor,
        r: Tensor,
        t: Tensor,
        timestep: int,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        task: Optional[str] = "relation_prediction",
    ) -> Tensor:
        """
        Denoising score-matching loss with noise-conditional score networks.
        """
        h_emb, r_emb, t_emb = (
            score_model.embedding_model.node_emb(h),
            score_model.embedding_model.rel_emb(r),
            score_model.embedding_model.node_emb(t),
        )
        if x is not None and self.feature_transform:
            h_emb += self.feature_transform(x)

        true_score = score_model(
            h_emb, r_emb, t_emb
        )  # simply do not pass in the timestep
        noisy_score = score_model(h_emb, r_emb, t_emb, timestep)
        return ((true_score - noisy_score) ** 2).mean()
