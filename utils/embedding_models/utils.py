from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.kge.loader import KGTripletLoader

from ..typing_utils import KG_Completion_Metrics


def evaluate_prediction_task(
    model: nn.Module,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor,
    batch_size: int,
    x: Optional[Tensor],
    k: Optional[List[int]] = [1, 3, 10],
    only_relation_prediction: Optional[bool] = False,
) -> KG_Completion_Metrics:
    """
    Evaluate the prediction task for a given model, head, relation, and tail indices.

    Args:
        model (nn.Module): The model to evaluate.
        head_index (Tensor): The tensor containing the head indices.
        rel_type (Tensor): The tensor containing the relation types.
        tail_index (Tensor): The tensor containing the tail indices.
        batch_size (int): The batch size to use for evaluation.
        x (Optional[Tensor]): Optional additional input tensor.
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
    if only_relation_prediction:
        relation_prediction_scores = compute_prediction_scores_vectorized(
            model,
            head_index,
            rel_type,
            tail_index,
            batch_size,
            x,
            only_relation_prediction,
        )

        relation_ranks = (
            relation_prediction_scores.argsort(descending=True, dim=1)  # type: ignore
            == rel_type.unsqueeze(1)
        ).nonzero(as_tuple=True)[1]

        relation_mean_rank = relation_ranks.float().mean().item()

        relation_mrr = (1.0 / (relation_ranks.float() + 1)).mean().item()

        relation_hits_at_k = {}

        if k:
            for k_val in k:
                relation_hits_at_k[k_val] = (
                    (relation_ranks < k_val).float().mean().item()
                )
        else:
            raise ValueError(f"bad k value: {k}")
        return (
            relation_mean_rank,
            relation_mrr,
            relation_hits_at_k,
        )
    else:
        (
            tail_prediction_scores,
            head_prediction_scores,
            relation_prediction_scores,
        ) = compute_prediction_scores_vectorized(
            model,
            head_index,
            rel_type,
            tail_index,
            batch_size,
            x,
            only_relation_prediction,
        )

        head_ranks = (
            head_prediction_scores.argsort(descending=True, dim=1)
            == head_index.unsqueeze(1)
        ).nonzero(as_tuple=True)[1]
        relation_ranks = (
            relation_prediction_scores.argsort(descending=True, dim=1)
            == rel_type.unsqueeze(1)
        ).nonzero(as_tuple=True)[1]
        tail_ranks = (
            tail_prediction_scores.argsort(descending=True, dim=1)
            == tail_index.unsqueeze(1)
        ).nonzero(as_tuple=True)[1]

        head_mean_rank = head_ranks.float().mean().item()
        relation_mean_rank = relation_ranks.float().mean().item()
        tail_mean_rank = tail_ranks.float().mean().item()

        head_mrr = (1.0 / (head_ranks.float() + 1)).mean().item()
        relation_mrr = (1.0 / (relation_ranks.float() + 1)).mean().item()
        tail_mrr = (1.0 / (tail_ranks.float() + 1)).mean().item()

        head_hits_at_k = {}
        relation_hits_at_k = {}
        tail_hits_at_k = {}

        if k:
            for k_val in k:
                head_hits_at_k[k_val] = (
                    (head_ranks < k_val).float().mean().item()
                )
                relation_hits_at_k[k_val] = (
                    (relation_ranks < k_val).float().mean().item()
                )
                tail_hits_at_k[k_val] = (
                    (tail_ranks < k_val).float().mean().item()
                )
        else:
            raise ValueError(f"bad k value: {k}")

        return (
            head_mean_rank,
            relation_mean_rank,
            tail_mean_rank,
            head_mrr,
            relation_mrr,
            tail_mrr,
            head_hits_at_k,
            relation_hits_at_k,
            tail_hits_at_k,
        )


def compute_prediction_scores_vectorized(
    model: nn.Module,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor,
    batch_size: int,
    x: Optional[Tensor],
    only_relation_prediction: Optional[bool],
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """
    Compute the prediction scores for the given model, head, relation, and tail indices.

    Args:
        model (nn.Module): The model to use for prediction.
        head_index (Tensor): The tensor containing the head indices.
        rel_type (Tensor): The tensor containing the relation types.
        tail_index (Tensor): The tensor containing the tail indices.
        batch_size (int): The batch size to use for computation.
        x (Optional[Tensor]): Optional additional input tensor.
        only_relation_prediction (bool): A bool to decide whether to perform head, relation, and tail prediction or just relation prediction only.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            A tuple containing the following tensors:
            - tail_prediction_scores: The scores for tail predictions.
            - head_prediction_scores: The scores for head predictions.
            - relation_prediction_scores: The scores for relation predictions.
    """
    if only_relation_prediction:

        all_scores_relation_prediction = []
        indices = torch.arange(model.num_relations, device=rel_type.device)
        for idx_batch in indices.split(batch_size):
            head_index_broadcasted = head_index.unsqueeze(1).repeat(
                1, idx_batch.size(0)
            )
            tail_index_broadcasted = tail_index.unsqueeze(1).repeat(
                1, idx_batch.size(0)
            )
            scores_relation_prediction = model(
                head_index_broadcasted, idx_batch, tail_index_broadcasted, x
            )
            all_scores_relation_prediction.append(scores_relation_prediction)

        return torch.cat(all_scores_relation_prediction, dim=1)

    else:
        all_scores_tail_prediction = []
        all_scores_head_prediction = []
        all_scores_relation_prediction = []
        indices = torch.arange(model.num_nodes, device=head_index.device)
        for idx_batch in indices.split(batch_size):
            rel_type_broadcasted = rel_type.unsqueeze(1).repeat(
                1, idx_batch.size(0)
            )
            tail_index_broadcasted = tail_index.unsqueeze(1).repeat(
                1, idx_batch.size(0)
            )
            scores_head_prediction = model(
                idx_batch, rel_type_broadcasted, tail_index_broadcasted, x
            )
            all_scores_head_prediction.append(scores_head_prediction)
        for idx_batch in indices.split(batch_size):
            head_index_broadcasted = head_index.unsqueeze(1).repeat(
                1, idx_batch.size(0)
            )
            rel_type_broadcasted = rel_type.unsqueeze(1).repeat(
                1, idx_batch.size(0)
            )
            scores_tail_prediction = model(
                head_index_broadcasted, rel_type_broadcasted, idx_batch, x
            )
            all_scores_tail_prediction.append(scores_tail_prediction)
        indices = torch.arange(model.num_relations, device=rel_type.device)
        for idx_batch in indices.split(batch_size):
            head_index_broadcasted = head_index.unsqueeze(1).repeat(
                1, idx_batch.size(0)
            )
            tail_index_broadcasted = tail_index.unsqueeze(1).repeat(
                1, idx_batch.size(0)
            )
            scores_relation_prediction = model(
                head_index_broadcasted, idx_batch, tail_index_broadcasted, x
            )
            all_scores_relation_prediction.append(scores_relation_prediction)

        return (
            torch.cat(all_scores_tail_prediction, dim=1),
            torch.cat(all_scores_head_prediction, dim=1),
            torch.cat(all_scores_relation_prediction, dim=1),
        )


def evaluate_classification_task(
    model: nn.Module,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor,
    x: Optional[Tensor],
    y: Optional[Tensor],
) -> float:
    """
    Evaluates the node classification task by computing the accuracy of the predictions.

    Args:
        model (nn.Module): The model to evaluate.
        head_index (Tensor): The head indices.
        rel_type (Tensor): The relation type indices.
        tail_index (Tensor): The tail indices - here they will be auxiliary node indices.
        x (Tensor, optional): Additional node features.
        y (Tensor, optional): The ground truth labels.

    Returns:
        float: The accuracy of the classification.
    """
    if model.aux_dict is None:
        raise ValueError("aux_dict isn't provided")

    scores = compute_classification_scores(
        model, head_index, rel_type, tail_index, x
    )
    aux_keys_tensor = torch.tensor(
        list(model.aux_dict.keys()), device=scores.device, dtype=torch.long
    )
    predicted_labels = aux_keys_tensor[scores.argmax(dim=0)]
    accuracy = (predicted_labels == tail_index).float().mean().item()
    return accuracy


def compute_classification_scores(
    model: nn.Module,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor,
    x: Optional[Tensor],
) -> Tensor:
    """
    Computes classification scores for each node in the graph.

    Args:
        model (nn.Module): The model used for scoring.
        head_index (Tensor): The head indices.
        rel_type (Tensor): The relation type indices.
        tail_index (Tensor): The tail indices.
        x (Tensor, optional): Additional node features.

    Returns:
        Tensor: A tensor containing the classification scores.
    """
    aux_indices = torch.tensor(
        list(model.aux_dict.keys()), dtype=torch.long, device=tail_index.device
    )
    scores = torch.stack(
        [
            model(head_index, rel_type, aux_idx.expand_as(tail_index), x)
            for aux_idx in aux_indices
        ]
    )
    return scores


class CustomKGTripletLoader(KGTripletLoader):
    def __init__(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        **kwargs,
    ):
        """
        Initializes the CustomKGTripletLoader with the ability to handle additional data attributes.

        Args:
            head_index (Tensor): Tensor of head indices in the triplets.
            rel_type (Tensor): Tensor of relation types in the triplets.
            tail_index (Tensor): Tensor of tail indices in the triplets.
            **kwargs: Additional keyword arguments that may include 'x' and 'y' for node features and labels.

        The 'x' and 'y' parameters are popped from the kwargs to prevent them from being passed to the superclass
        constructor, which does not recognize these parameters.
        """
        # Store the additional data
        self.x = kwargs.pop("x", None)
        self.y = kwargs.pop("y", None)
        # Initialize the superclass with the remaining kwargs
        super().__init__(head_index, rel_type, tail_index, **kwargs)

    def sample(self, index: List[int]) -> Dict[str, Tensor]:
        """
        Samples a batch of data based on the provided indices and adds additional features and labels if available.

        Args:
            index (List[int]): List of indices specifying which samples to include in the batch.

        Returns:
            Dict[str, Tensor]: A dictionary containing tensors for 'head_index', 'rel_type', 'tail_index',
            and optionally 'x' and 'y' if they were provided during initialization.
        """
        # Convert list of indices to a tensor of the appropriate device
        index = torch.tensor(index, device=self.head_index.device)

        # Extract the corresponding elements for head, relation, and tail
        head_index = self.head_index[index]
        rel_type = self.rel_type[index]
        tail_index = self.tail_index[index]

        # Prepare the result dictionary with mandatory elements
        result = {
            "head_index": head_index,
            "rel_type": rel_type,
            "tail_index": tail_index,
        }

        # Conditionally add x and y to the result if they are not None
        if self.x is not None:
            result["x"] = self.x[index]
        if self.y is not None:
            result["y"] = self.y[index]

        return result
