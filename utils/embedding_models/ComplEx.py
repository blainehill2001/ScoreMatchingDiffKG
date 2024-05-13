from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.nn import ComplEx as BaseComplEx

from .utils import CustomKGTripletLoader
from .utils import evaluate_classification_task
from .utils import evaluate_prediction_task
from .utils import random_sample


class CustomComplEx(BaseComplEx):
    """
    A custom implementation of the ComplEx model for knowledge graph embeddings,
    extended to support additional features and tasks.

    Attributes:
        feature_transform_re (nn.Module): Linear transformation for real part of node features (if applicable).
        feature_transform_im (nn.Module): Linear transformation for imaginary part of node features (if applicable).
        task (str): The specific task for which the model is being used.
        aux_dict (dict): Auxiliary dictionary for additional task-specific data.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the CustomComplEx model.

        Args:
            *args: Variable length argument list for base ComplEx model.
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

        if feature_dim and embedding_dim:
            self.feature_transform_re = nn.Linear(feature_dim, embedding_dim)
            self.feature_transform_im = nn.Linear(feature_dim, embedding_dim)
        else:
            self.feature_transform_re, self.feature_transform_im = None, None

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
        Forward pass of the CustomComplEx model.

        Args:
            head_index (Tensor): Indices of head entities.
            rel_type (Tensor): Indices of relation types.
            tail_index (Tensor): Indices of tail entities.
            x (Tensor, optional): Additional node features.
            y (Tensor, optional): Node labels (unused, for API consistency).

        Returns:
            Tensor: Scores representing the model's predictions.
        """
        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        rel_re = self.rel_emb(rel_type)
        rel_im = self.rel_emb_im(rel_type)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)

        if x is not None:
            # Transform node features and add to head embeddings
            head_re += self.feature_transform_re(x)
            head_im += self.feature_transform_im(x)

        # Compute the scoring function components
        score_re_re = head_re * rel_re * tail_re
        score_im_re = head_im * rel_re * tail_im
        score_re_im = head_re * rel_im * tail_im
        score_im_im = head_im * rel_im * tail_re

        # Sum up the components to get the final score
        return (score_re_re + score_im_re + score_re_im - score_im_im).sum(
            dim=-1
        )

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
            *self.random_sample(
                head_index, rel_type, tail_index, task, aux_dict
            ),
            x,
            y,
        )
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)

        return F.binary_cross_entropy_with_logits(scores, target)

    def loader(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        **kwargs,
    ) -> DataLoader:
        """
        Creates a data loader for training or evaluation that generates batches of triplets.

        This method wraps the CustomKGTripletLoader to handle the specific needs of the CustomComplEx model,
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

    def random_sample(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        task: Optional[str],
        aux_dict: Optional[Dict] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generates a random negative sample for training.

        Args:
            head_index (Tensor): Indices of head entities.
            rel_type (Tensor): Indices of relation types.
            tail_index (Tensor): Indices of tail entities.
            task (str, optional): The specific task for which the model is being used.
            aux_dict (dict, optional): Auxiliary dictionary for additional data.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Indices of head, relation, and tail for the negative sample.
        """
        return random_sample(
            model=self,
            head_index=head_index,
            rel_type=rel_type,
            tail_index=tail_index,
            task=task,
            aux_dict=aux_dict,
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
        k: List[int] = [1, 3, 10],
        task: str = "relation_prediction",  # default to link prediction
    ) -> Union[float, Tuple[float, float, Dict[int, float]]]:
        """
        Evaluates the model on a test set with specified parameters.

        Args:
            head_index (Tensor): The head indices.
            rel_type (Tensor): The relation type.
            tail_index (Tensor): The tail indices.
            batch_size (int): The batch size to use for evaluating.
            x (Tensor, optional): Node features.
            y (Tensor, optional): Node labels.
            k (List[int]): The `k` in Hits @ `k`.
            task (str, optional): The task to perform ("relation_prediction", "head_prediction", "tail_prediction", "node_classification").

        Returns:
            Union[float, Tuple[float, float, float]]: Either a single float or a tuple of floats representing evaluation metrics.
        """
        if task in [
            "relation_prediction",
            "head_prediction",
            "tail_prediction",
        ]:
            return self.evaluate_prediction_task(
                head_index, rel_type, tail_index, batch_size, x, k, task
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
        k: List[int] = [1, 3, 10],
        task: str = "relation_prediction",
    ) -> Tuple[float, float, Dict[int, float]]:
        """
        Helper function to evaluate prediction tasks.

        Args:
            head_index (Tensor): The head indices.
            rel_type (Tensor): The relation type.
            tail_index (Tensor): The tail indices.
            batch_size (int): The batch size to use for evaluating.
            x (Tensor, optional): Node features.
            k (Union[int, List[int]]): The `k` in Hits @ `k`.
            task (str): The task to perform.

        Returns:
            Tuple[float, float, Dict[int, float]]: A tuple containing evaluation metrics (mean rank, MRR, hits@k).
        """
        return evaluate_prediction_task(
            self, head_index, rel_type, tail_index, batch_size, x, k, task
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
