from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.nn import DistMult as BaseDistMult

from ..typing_utils import KG_Completion_Metrics
from .utils import CustomKGTripletLoader
from .utils import evaluate_classification_task
from .utils import evaluate_prediction_task


class CustomDistMult(BaseDistMult):
    """
    A custom implementation of the DistMult model for knowledge graph embeddings,
    extended to support additional features and tasks.

    Attributes:
        feature_transform_re (nn.Module): Linear transformation for real part of node features (if applicable).
        feature_transform_im (nn.Module): Linear transformation for imaginary part of node features (if applicable).
        task (str): The specific task for which the model is being used.
        aux_dict (dict): Auxiliary dictionary for additional task-specific data.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the CustomDistMult model.

        Args:
            *args: Variable length argument list for base DistMult model.
            **kwargs: Arbitrary keyword arguments. Important custom keywords are:
                - head_node_feature_dim (int): Dimensionality of the head node features.
                - task (str): Specific task for the model usage.
                - aux_dict (dict): Auxiliary dictionary for additional data.
                - hidden_channels (int): Dimensionality for the embedding space.
        """
        feature_dim = kwargs.pop("head_node_feature_dim", None)
        task = kwargs.pop("task", None)
        aux_dict = kwargs.pop("aux_dict", None)
        super().__init__(*args, **kwargs)
        embedding_dim = kwargs.get("hidden_channels", None)

        self.feature_transform_re = (
            nn.Linear(feature_dim, embedding_dim)
            if feature_dim and embedding_dim
            else None
        )

        self.task = task
        self.aux_dict = (
            {int(key): value for key, value in aux_dict.items()}
            if aux_dict is not None
            else None
        )

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the CustomDistMult model.

        Args:
            head_index (Tensor): Indices of head entities.
            rel_type (Tensor): Indices of relation types.
            tail_index (Tensor): Indices of tail entities.
            x (Tensor, optional): Additional node features.
            y (Tensor, optional): Node labels (unused, for API consistency).

        Returns:
            Tensor: Scores representing the model's predictions.
        """
        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)

        if x is not None:
            # Transform node features and add to head embeddings
            head += self.feature_transform_re(x)

        # Element-wise multiplication of embeddings and sum reduction
        return (head * rel * tail).sum(dim=-1)

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        task: Optional[str] = "relation_prediction",
        aux_dict: Optional[Dict] = None,
    ) -> Tensor:
        """
        Computes the loss for a batch of triples.

        Args:
            head_index (Tensor): Indices of head entities.
            rel_type (Tensor): Indices of relation types.
            tail_index (Tensor): Indices of tail entities.
            x (Tensor, optional): Additional node features.
            y (Tensor, optional): Node labels (unused, for API consistency).
            task (str, optional): The specific task for which the model is being used.
            aux_dict (dict, optional): Auxiliary dictionary for additional data.

        Returns:
            Tensor: The computed loss value.
        """
        pos_score = self(head_index, rel_type, tail_index, x, y)
        neg_score = self(
            *self.random_sample(head_index, rel_type, tail_index),
            x,
            y,
        )

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )

    def loader(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        **kwargs,
    ) -> DataLoader:
        """
        Creates a data loader for training or evaluation that generates batches of triplets.

        This method wraps the CustomKGTripletLoader to handle the specific needs of the CustomDistMult model,
        such as handling additional features and tasks.

        Args:
            head_index (Tensor): Tensor of head indices in the triplets.
            rel_type (Tensor): Tensor of relation types in the triplets.
            tail_index (Tensor): Tensor of tail indices in the triplets.
            **kwargs: Additional keyword arguments that are passed to the CustomKGTripletLoader.

        Returns:
            DataLoader: A data loader that yields batches of triplets for training or evaluation.
        """
        return CustomKGTripletLoader(
            head_index, rel_type, tail_index, **kwargs
        )

    @torch.no_grad()
    def test(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        batch_size: int,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        k: Optional[List[int]] = [1, 3, 10],
        task: Optional[str] = "kg_completion",
        only_relation_prediction: Optional[bool] = False,
    ) -> Union[KG_Completion_Metrics, float]:
        """
        Evaluates the model on a test set with specified parameters.

        Args:
            head_index (Tensor): The head indices.
            rel_type (Tensor): The relation type.
            tail_index (Tensor): The tail indices.
            batch_size (int): The batch size to use for evaluating.
            x (Tensor, optional): Node features.
            y (Tensor, optional): Node labels.
            k (List[int], optional): The `k` in Hits @ `k`.
            task (str, optional): The task to perform ("relation_prediction", "head_prediction", "tail_prediction", "node_classification").
            only_relation_prediction (bool, optional): A bool to decide whether to perform head, relation, and tail prediction or just relation prediction only.

        Returns:
            Union[KG_Completion_Metrics, float]:
            Either a single float or a tuple of floats and dict(s) representing evaluation metrics.
        """
        if task == "kg_completion":
            return self.evaluate_prediction_task(
                head_index,
                rel_type,
                tail_index,
                batch_size,
                x,
                k,
                only_relation_prediction,
            )
        elif task == "node_classification":
            return self.evaluate_classification_task(
                head_index, rel_type, tail_index, x, y
            )
        else:
            raise ValueError(f"Unsupported task type: {task}")

    def evaluate_prediction_task(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        batch_size: int,
        x: Optional[Tensor],
        k: Optional[List[int]] = [1, 3, 10],
        only_relation_prediction: Optional[bool] = False,
    ) -> KG_Completion_Metrics:
        """
        Helper function to evaluate prediction tasks.

        Args:
            head_index (Tensor): The head indices.
            rel_type (Tensor): The relation type.
            tail_index (Tensor): The tail indices.
            batch_size (int): The batch size to use for evaluating.
            x (Tensor, optional): Node features.
            k (List[int], optional): The `k` in Hits @ `k`.
            only_relation_prediction (bool, optional): A bool to decide whether to perform head, relation, and tail prediction or just relation prediction only.

        Returns:
            KG_Completion_Metrics:
                A tuple containing the either of the following values:
                - relation_mean_rank: The mean rank for relation predictions.
                - relation_mrr: The mean reciprocal rank for relation predictions.
                - relation_hits_at_k: A dictionary containing hits@k for relation predictions.
                or
                - head_mean_rank: The mean rank for head predictions.
                - relation_mean_rank: The mean rank for relation predictions.
                - tail_mean_rank: The mean rank for tail predictions.
                - head_mrr: The mean reciprocal rank for head predictions.
                - relation_mrr: The mean reciprocal rank for relation predictions.
                - tail_mrr: The mean reciprocal rank for tail predictions.
                - head_hits_at_k: A dictionary containing hits@k for head predictions.
                - relation_hits_at_k: A dictionary containing hits@k for relation predictions.
                - tail_hits_at_k: A dictionary containing hits@k for tail predictions.
        """
        return evaluate_prediction_task(
            self,
            head_index,
            rel_type,
            tail_index,
            batch_size,
            x,
            k,
            only_relation_prediction,
        )

    def evaluate_classification_task(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        x: Optional[Tensor],
        y: Optional[Tensor],
    ) -> float:
        """
        Helper function to evaluate node classification tasks.

        Args:
            head_index (Tensor): The head indices.
            rel_type (Tensor): The relation type.
            tail_index (Tensor): The tail indices.
            x (Tensor, optional): Node features.
            y (Tensor, optional): Node labels.

        Returns:
            float: The accuracy of the classification.
        """
        return evaluate_classification_task(
            self, head_index, rel_type, tail_index, x, y
        )
