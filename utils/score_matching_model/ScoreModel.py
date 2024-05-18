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

from ..typing_utils import KG_Completion_Metrics


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

        self.node_emb_dim = self.entity_embeddings_weights.shape[-1]
        self.rel_emb_dim = self.relation_embeddings_weights.shape[-1]
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
        if self.similarity_metric == "l2":
            distance = torch.linalg.norm(h_emb + r_emb - t_emb, dim=-1)
        elif self.similarity_metric == "cosine":
            distance = -F.cosine_similarity(h_emb + r_emb, t_emb, dim=-1)
        elif self.similarity_metric == "metric_tensor":
            score_grad_h, score_grad_r, score_grad_t = self.score_function(
                h_emb, r_emb, t_emb
            )
            (
                local_metric_tensor_h,
                local_metric_tensor_r,
                local_metric_tensor_t,
            ) = self.local_metric_tensor(
                score_grad_h, score_grad_r, score_grad_t
            )
            local_metric_tensor = (
                local_metric_tensor_h
                + local_metric_tensor_r
                + local_metric_tensor_t
            ) / 3
            avg_metric_tensor = (
                local_metric_tensor
                + torch.eye(
                    local_metric_tensor.shape[-1],
                    device=local_metric_tensor.device,
                )
            ) / 2
            distance = torch.sum(
                (h_emb + r_emb - t_emb) ** 2 * avg_metric_tensor, dim=-1
            )
        else:
            raise ValueError(
                f"Invalid similarity metric: {self.similarity_metric}. "
                "Supported metrics are 'l2', 'cosine', and 'metric_tensor'."
            )

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

    def reverse_sde_prediction(
        self, emb1: Tensor, emb2: Tensor, task: str
    ) -> Tensor:
        """
        Refine the embeddings using reverse SDE for better KG Completion.

        Args:
            emb1: Embeddings of the first entity/relation.
            emb2: Embeddings of the second entity/relation.
            task: Specific task to perform - "relation_prediction", "head_prediction", "tail_prediction", or "node_classification".
        Returns:
            Tensor: Refined embeddings.
        """
        device = emb1.device

        # Initialize with random noise based on the task
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
                elif task == "tail_prediction":
                    score = self(emb1, emb2, pred_emb, t)
                else:  # task == "node_classification"
                    score = self(emb1, emb2, pred_emb, t)
                grad_pred_emb = torch.autograd.grad(
                    score.sum(), pred_emb, create_graph=True
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
        task: str = "kg_completion",
        only_relation_prediction: bool = False,
    ) -> Union[KG_Completion_Metrics, float]:
        """
        Test the model on a given task.

        Args:
            h: Head entity tensor.
            r: Relation tensor.
            t: Tail entity tensor.
            x: Additional tensor for transformation.
            k: List of top-k values.
            task: Type of prediction task.
            only_relation_prediction: Bool to determine whether to compute all metrics or just relation prediction metrics for prediction task

        Returns:
            KG_Completion_Metrics: Evaluation results.
        """

        if self.task == "kg_completion":
            return self.evaluate_prediction_task(
                h, r, t, x, k, only_relation_prediction
            )
        elif task == "node_classification":
            return self.evaluate_classification_task(h, r, t, x)
        else:
            raise ValueError(f"Unsupported task type: {task}")

    def compute_prediction_metrics(
        self, ranks: Tensor, k: List[int]
    ) -> Tuple[float, float, Dict[int, float]]:
        """
        Compute evaluation metrics from ranks.

        Args:
            ranks: Tensor of ranks.
            k: List of top-k values.

        Returns:
            Tuple[float, float, Dict[int, float]]: Mean rank, MRR, and Hits@k.
        """
        mean_rank = ranks.float().mean().item()
        mrr = (1.0 / (ranks.float() + 1)).mean().item()
        hits_at_k = {
            k_val: (ranks < k_val).float().mean().item() for k_val in k
        }
        return mean_rank, mrr, hits_at_k

    def evaluate_prediction_task(
        self,
        h: Tensor,
        r: Tensor,
        t: Tensor,
        x: Optional[Tensor],
        k: List[int] = [1, 3, 10],
        only_relation_prediction: bool = False,
    ) -> KG_Completion_Metrics:
        """
        Evaluate the model on a prediction task.

        Args:
            h: Head entity tensor.
            r: Relation tensor.
            t: Tail entity tensor.
            x: Additional tensor for transformation.
            k: List of top-k values.
            only_relation_prediction: Bool to determine whether to compute all metrics or just relation prediction metrics.

        Returns:
            KG_Completion_Metrics: Evaluation metrics.
        """
        h_emb = self.embedding_model.node_emb(h)
        r_emb = self.embedding_model.rel_emb(r)
        t_emb = self.embedding_model.node_emb(t)
        embedding_weights_r = self.embedding_model.rel_emb.weight.detach()
        embedding_weights_h_t = self.embedding_model.node_emb.weight.detach()

        if only_relation_prediction:
            sorted_values_r, sorted_indices_r = (
                self.calculate_similarity_and_sort(
                    h_emb,
                    r_emb,
                    t_emb,
                    embedding_weights_r,
                    "relation_prediction",
                )
            )
            ranks_r = (sorted_indices_r == r.unsqueeze(1)).nonzero(
                as_tuple=True
            )[1]
            mean_rank_r, mrr_r, hits_at_k_r = self.compute_prediction_metrics(
                ranks_r, k
            )
            return mean_rank_r, mrr_r, hits_at_k_r

        sorted_values_h, sorted_indices_h = self.calculate_similarity_and_sort(
            h_emb, r_emb, t_emb, embedding_weights_h_t, "head_prediction"
        )
        sorted_values_r, sorted_indices_r = self.calculate_similarity_and_sort(
            h_emb, r_emb, t_emb, embedding_weights_r, "relation_prediction"
        )
        sorted_values_t, sorted_indices_t = self.calculate_similarity_and_sort(
            h_emb, r_emb, t_emb, embedding_weights_h_t, "tail_prediction"
        )

        ranks_h = (sorted_indices_h == h.unsqueeze(1)).nonzero(as_tuple=True)[
            1
        ]
        ranks_r = (sorted_indices_r == r.unsqueeze(1)).nonzero(as_tuple=True)[
            1
        ]
        ranks_t = (sorted_indices_t == t.unsqueeze(1)).nonzero(as_tuple=True)[
            1
        ]

        mean_rank_h, mrr_h, hits_at_k_h = self.compute_prediction_metrics(
            ranks_h, k
        )
        mean_rank_r, mrr_r, hits_at_k_r = self.compute_prediction_metrics(
            ranks_r, k
        )
        mean_rank_t, mrr_t, hits_at_k_t = self.compute_prediction_metrics(
            ranks_t, k
        )

        return (
            mean_rank_h,
            mean_rank_r,
            mean_rank_t,
            mrr_h,
            mrr_r,
            mrr_t,
            hits_at_k_h,
            hits_at_k_r,
            hits_at_k_t,
        )

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

    def calculate_similarity_and_sort(
        self,
        h_emb: Tensor,
        r_emb: Tensor,
        t_emb: Tensor,
        embedding_weights: Tensor,
        task: str,
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculate the similarity between refined embeddings and the embedding weights, then sort the results.

        This method supports different similarity metrics ('l2', 'cosine', 'metric_tensor') and tasks
        ('relation_prediction', 'head_prediction', 'tail_prediction'). Depending on the task, it refines
        the appropriate embeddings using reverse SDE prediction and then calculates the similarity
        based on the specified metric. The results are sorted in ascending order for distances or
        descending order for similarities.

        Args:
            h_emb (Tensor): Embedding of the head entity.
            r_emb (Tensor): Embedding of the relation.
            t_emb (Tensor): Embedding of the tail entity.
            embedding_weights (Tensor): Embedding weights against which to compare.
            task (str): The task type, which determines which embeddings to refine and compare.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the sorted values and their corresponding indices.
        """
        if self.similarity_metric == "l2":
            if task == "relation_prediction":
                refined_emb = self.reverse_sde_prediction(h_emb, t_emb, task)
            elif task == "head_prediction":
                refined_emb = self.reverse_sde_prediction(r_emb, t_emb, task)
            else:  # tail_prediction
                refined_emb = self.reverse_sde_prediction(h_emb, r_emb, task)
            dist = torch.norm(
                refined_emb.unsqueeze(1) - embedding_weights.unsqueeze(0),
                p=2,
                dim=-1,
            )
            sorted_values, sorted_indices = torch.sort(dist, dim=1)
        elif self.similarity_metric == "cosine":
            if task == "relation_prediction":
                refined_emb = self.reverse_sde_prediction(h_emb, t_emb, task)
            elif task == "head_prediction":
                refined_emb = self.reverse_sde_prediction(r_emb, t_emb, task)
            else:  # tail_prediction
                refined_emb = self.reverse_sde_prediction(h_emb, r_emb, task)
            similarity = F.cosine_similarity(
                refined_emb.unsqueeze(1),
                embedding_weights.unsqueeze(0),
                dim=-1,
            )
            sorted_values, sorted_indices = torch.sort(
                similarity, dim=1, descending=True
            )
        elif self.similarity_metric == "metric_tensor":
            score_grad_h, score_grad_r, score_grad_t = self.score_function(
                h_emb, r_emb, t_emb
            )
            (
                local_metric_tensor_h,
                local_metric_tensor_r,
                local_metric_tensor_t,
            ) = self.local_metric_tensor(
                score_grad_h, score_grad_r, score_grad_t
            )
            if task == "relation_prediction":
                refined_emb = self.reverse_sde_prediction(h_emb, t_emb, task)
                embedding_weights = self.relation_embeddings_weights
                local_metric_tensor = local_metric_tensor_r
            elif task == "head_prediction":
                refined_emb = self.reverse_sde_prediction(r_emb, t_emb, task)
                embedding_weights = self.entity_embeddings_weights
                local_metric_tensor = local_metric_tensor_h
            else:  # tail_prediction
                refined_emb = self.reverse_sde_prediction(h_emb, r_emb, task)
                embedding_weights = self.entity_embeddings_weights
                local_metric_tensor = local_metric_tensor_t
            avg_metric_tensor = (
                local_metric_tensor
                + torch.eye(
                    local_metric_tensor.shape[-1],
                    device=local_metric_tensor.device,
                )
            ) / 2
            dist = torch.sum(
                (refined_emb.unsqueeze(1) - embedding_weights.unsqueeze(0))
                ** 2
                * avg_metric_tensor,
                dim=-1,
            )
            sorted_values, sorted_indices = torch.sort(dist, dim=1)
        else:
            raise ValueError(
                f"Invalid similarity metric: {self.similarity_metric}. "
                "Supported metrics are 'l2', 'cosine', and 'metric_tensor'."
            )

        return sorted_values, sorted_indices
