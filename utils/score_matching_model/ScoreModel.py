import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScoreModel(nn.Module):
    def __init__(
        self,
        embedding_model,
        score_model_hidden_dim: int = 512,
        num_sde_timesteps: int = 20,
        similarity_metric: str = "l2",
        task: str = "relation_prediction",
        aux_dict: Optional[Dict] = None,
    ):
        """
        Initialize the ScoreModel.

        Args:
            embedding_model: The embedding model used for scoring.
            score_model_hidden_dim: The hidden dimension of the score model.
            num_sde_timesteps: Number of SDE timesteps.
            similarity_metric: The similarity metric used for evaluation.
            task: The task type for the model.
            aux_dict: Auxiliary dictionary for additional information.
        """
        super().__init__()
        self.embedding_model = embedding_model
        # Get entity and relation embeddings from embedding model
        with torch.no_grad():
            entity_embeddings_weights = (
                embedding_model.node_emb.weight.detach()
            )
            relation_embeddings_weights = (
                embedding_model.rel_emb.weight.detach()
            )

            # handle both real and imaginary node and relation embeddings
            if hasattr(embedding_model, "node_emb_im") and hasattr(
                embedding_model, "feature_transform_im"
            ):  # occurs with RotatE, ComplEx
                entity_embeddings_weights_im = (
                    embedding_model.node_emb_im.weight.detach()
                )
                entity_embeddings_weights = torch.cat(
                    [entity_embeddings_weights, entity_embeddings_weights_im],
                    dim=-1,
                )

            if hasattr(embedding_model, "rel_emb_im"):  # occurs with ComplEx
                relation_embeddings_weights_im = (
                    embedding_model.rel_emb_im.weight.detach()
                )
                relation_embeddings_weights = torch.cat(
                    [
                        relation_embeddings_weights,
                        relation_embeddings_weights_im,
                    ],
                    dim=-1,
                )

            if hasattr(embedding_model, "feature_transform_re") and hasattr(
                embedding_model, "feature_transform_im"
            ):
                in_features_1 = (
                    embedding_model.feature_transform_re.in_features
                )
                out_features_1 = (
                    embedding_model.feature_transform_re.out_features
                )
                in_features_2 = (
                    embedding_model.feature_transform_im.in_features
                )
                out_features_2 = (
                    embedding_model.feature_transform_im.out_features
                )

                combined_in_features = in_features_1 + in_features_2
                combined_out_features = out_features_1 + out_features_2

                feature_transform = nn.Linear(
                    combined_in_features, combined_out_features
                )

                # Copy the weights from the original layers to the new combined layer
                feature_transform.weight.data[
                    :in_features_1, :out_features_1
                ] = embedding_model.feature_transform_re.weight.data
                feature_transform.weight.data[
                    in_features_1:, out_features_1:
                ] = embedding_model.feature_transform_im.weight.data
                feature_transform.bias.data[:out_features_1] = (
                    embedding_model.feature_transform_re.bias.data
                )
                feature_transform.bias.data[out_features_1:] = (
                    embedding_model.feature_transform_im.bias.data
                )

            else:
                feature_transform = embedding_model.feature_transform_re

        self.entity_embeddings_weights = entity_embeddings_weights
        self.relation_embeddings_weights = relation_embeddings_weights
        self.feature_transform = feature_transform

        self.node_emb_dim = (self.entity_embeddings_weights.shape[-1],)
        self.rel_emb_dim = (self.relation_embeddings_weights.shape[-1],)
        self.score_net = nn.Sequential(
            nn.Linear(
                self.entity_embeddings_weights.shape[-1]
                + self.relation_embeddings_weights.shape[-1],
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
        self.num_sde_timesteps = num_sde_timesteps
        self.similarity_metric = similarity_metric
        self.task = task
        self.aux_dict = aux_dict

    def forward(
        self,
        h_emb: Tensor,
        r_emb: Tensor,
        t_emb: Tensor,
        timestep: Optional[int] = None,
    ) -> Tensor:
        """
        Forward pass of the model.

        Args:
            h_emb: Embedding of the head entity.
            r_emb: Embedding of the relation.
            t_emb: Embedding of the tail entity.
            timestep: Current timestep for SDE.

        Returns:
            Tensor: The calculated score.
        """
        # Implement your desired distance measure here (e.g., L2 distance)
        distance = torch.linalg.norm(
            h_emb + r_emb - t_emb, dim=-1
        )  # semantic meaning only (no graph structural information was used to calculate distance)

        # Gradually increase the weight of the distance term during SDE steps
        weight = 0.0  # No weight increase if timestep is None
        if timestep is not None:

            def sigmoid(x):
                return 1 / (1 + math.exp(-x))

            weight = sigmoid(timestep / (self.num_sde_timesteps - 1))
        score = weight * distance
        return score

    def loss(
        self,
        h: Tensor,
        r: Tensor,
        t: Tensor,
        timestep: int,
        x: Optional[Tensor] = None,
        task: Optional[str] = "relation_prediction",
    ) -> Tensor:
        """
        Calculate the denoising score-matching loss.

        Args:
            h: Head entity tensor.
            r: Relation tensor.
            t: Tail entity tensor.
            timestep: Current timestep.
            x: Additional tensor for transformation.
            task: Type of prediction task.

        Returns:
            Tensor: The calculated loss.
        """
        h_emb, r_emb, t_emb = (
            self.embedding_model.node_emb(h),
            self.embedding_model.rel_emb(r),
            self.embedding_model.node_emb(t),
        )
        if isinstance(x, torch.Tensor) and isinstance(
            self.feature_transform, torch.Tensor
        ):
            h_emb += torch.matmul(x, self.feature_transform)

        true_score = self(
            h_emb, r_emb, t_emb
        )  # simply do not pass in the timestep
        noisy_score = self(h_emb, r_emb, t_emb, timestep)
        return ((true_score - noisy_score) ** 2).mean()

    def reverse_sde_prediction(self, emb1: Tensor, emb2: Tensor) -> Tensor:
        """
        Refine the embeddings using reverse SDE for better link prediction.

        Args:
            emb1: Embeddings of the first entity/relation.
            emb2: Embeddings of the second entity/relation.

        Returns:
            Tensor: Refined embeddings.
        """
        task = self.task
        device = emb1.device

        # Initialize with random noise
        if task == "relation_prediction":
            pred_emb = torch.randn(
                emb1.size(0), self.rel_emb_dim, device=device
            )
        else:
            pred_emb = torch.randn(
                emb1.size(0), self.node_emb_dim, device=device
            )

        # Define the time steps for the reverse SDE process
        t_steps = torch.linspace(
            self.num_sde_timesteps - 1, 0, self.num_sde_timesteps
        ).to(device)

        # Define the SDE function for the reverse process
        def sde_func(t, pred_emb):
            with torch.enable_grad():
                pred_emb.requires_grad_(True)
                if task == "relation_prediction":
                    score = self(emb1, pred_emb, emb2, t)
                elif task == "head_prediction":
                    score = self(pred_emb, emb1, emb2, t)
                else:  # task in ["tail_prediction", "node_classification"]
                    score = self(emb1, emb2, pred_emb, t)
                grad_pred_emb = torch.autograd.grad(
                    score.sum(dim=0), pred_emb, create_graph=True
                )[0]
                return -grad_pred_emb

        # Reverse direction
        with torch.no_grad():
            for t in reversed(t_steps):
                pred_emb = sde_func(t, pred_emb)

        return pred_emb  # returns predictions for the specified entity type of shape (batch_size, emb_size)

    @torch.no_grad()
    def test(
        self,
        h: Tensor,
        r: Tensor,
        t: Tensor,
        x: Optional[Tensor] = None,
        k: List[int] = [1, 3, 10],
        task: str = "relation_prediction",
    ) -> Union[float, Tuple[float, float, Dict[int, float]]]:
        """
        Test the model on a given task.

        Args:
            h: Head entity tensor.
            r: Relation tensor.
            t: Tail entity tensor.
            x: Additional tensor for transformation.
            k: List of top-k values.
            task: Type of prediction task.

        Returns:
            Union[float, Tuple[float, float, Dict[int, float]]]: Evaluation results.
        """

        if self.task in [
            "relation_prediction",
            "head_prediction",
            "tail_prediction",
        ]:
            return self.evaluate_prediction_task(h, r, t, x, k)
        elif task == "node_classification":
            return self.evaluate_classification_task(h, r, t, x)
        else:
            raise ValueError(f"Unsupported task type: {task}")

    def evaluate_prediction_task(
        self,
        h: Tensor,
        r: Tensor,
        t: Tensor,
        x: Optional[Tensor],
        k: List[int] = [1, 3, 10],
    ) -> Tuple[float, float, Dict[int, float]]:
        """
        Evaluate the model on a prediction task.

        Args:
            h: Head entity tensor.
            r: Relation tensor.
            t: Tail entity tensor.
            x: Additional tensor for transformation.
            k: List of top-k values.

        Returns:
            Tuple[float, float, Dict[int, float]]: Evaluation metrics.
        """

        h_emb = self.embedding_model.node_emb(h)
        r_emb = self.embedding_model.rel_emb(r)
        t_emb = self.embedding_model.node_emb(t)

        if self.task == "relation_prediction":
            refined_emb = self.reverse_sde_prediction(h_emb, t_emb)
            embedding_weights = self.embedding_model.rel_emb.weight.detach()
            ground_truth = r
        elif self.task == "head_prediction":
            refined_emb = self.reverse_sde_prediction(r_emb, t_emb)
            embedding_weights = self.embedding_model.node_emb.weight.detach()
            ground_truth = h
        elif self.task == "tail_prediction":
            refined_emb = self.reverse_sde_prediction(h_emb, r_emb)
            embedding_weights = self.embedding_model.node_emb.weight.detach()
            ground_truth = t

        sorted_values, sorted_indices = self.calculate_similarity_and_sort(
            refined_emb, embedding_weights
        )

        # Calculate mean rank of the ground truth and MRR
        ranks = (sorted_indices == ground_truth.unsqueeze(1)).nonzero(
            as_tuple=True
        )[1] + 1
        mean_rank = ranks.float().mean().item()
        mrr = (1.0 / (ranks.float() + 1)).mean().item()

        # Check if true relation is within the top K predictions
        hits_at_k = {}
        if isinstance(k, int):
            k_values = [k]
        else:
            k_values = k
        for k_val in k_values:
            hits_at_k[k_val] = (ranks < k_val).float().mean().item()

        return mean_rank, mrr, hits_at_k

    def calculate_similarity_and_sort(
        self, refined_emb: Tensor, embedding_weights: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculate similarity and sort the embeddings.

        Args:
            refined_emb: Refined embeddings.
            embedding_weights: Embedding weights.

        Returns:
            Tuple[Tensor, Tensor]: Sorted values and indices.
        """
        if self.similarity_metric == "cosine":
            similarity = F.cosine_similarity(
                refined_emb.unsqueeze(1),
                embedding_weights.unsqueeze(0),
                dim=-1,
            )
        elif self.similarity_metric == "l2":
            dist = torch.norm(
                refined_emb.unsqueeze(1) - embedding_weights.unsqueeze(0),
                p=2,
                dim=-1,
            )
        else:
            raise NotImplementedError(
                f"Haven't implemented a similarity metric yet for {self.similarity_metric}"
            )

        if self.similarity_metric == "cosine":
            sorted_values, sorted_indices = torch.sort(
                similarity, dim=1, descending=True
            )
        elif self.similarity_metric == "l2":
            sorted_values, sorted_indices = torch.sort(dist, dim=1)

        return sorted_values, sorted_indices

    def evaluate_classification_task(
        self,
        h: Tensor,
        r: Tensor,
        t: Tensor,
        x: Optional[Tensor],
    ) -> float:
        """
        Evaluate the model on a classification task.

        Args:
            h: Head entity tensor.
            r: Relation tensor.
            t: Tail entity tensor.
            x: Additional tensor for transformation.

        Returns:
            float: Accuracy of the classification task.
        """
        if self.aux_dict is None:
            raise ValueError("aux_dict isn't provided")

        h_emb, r_emb, t_emb = (
            self.embedding_model.node_emb(h),
            self.embedding_model.rel_emb(r),
            self.embedding_model.node_emb(t),
        )

        if isinstance(x, torch.Tensor) and isinstance(
            self.feature_transform, torch.Tensor
        ):
            h_emb += torch.matmul(x, self.feature_transform)

        aux_indices_emb = self.embedding_model.node_emb(
            torch.tensor(
                list(self.aux_dict.keys()),
                dtype=torch.long,
                device=t_emb.device,
            )
        )
        scores = torch.stack(
            [
                self(h_emb, r_emb, aux_idx_emb.expand_as(t_emb))
                for aux_idx_emb in aux_indices_emb
            ]
        )
        aux_keys_tensor = torch.tensor(
            list(self.aux_dict.keys()), device=scores.device, dtype=torch.long
        )
        predicted_labels = aux_keys_tensor[scores.argmax(dim=0)]
        accuracy = (predicted_labels == t).float().mean().item()
        return accuracy
