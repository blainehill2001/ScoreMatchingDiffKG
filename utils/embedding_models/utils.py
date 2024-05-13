from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from torch import Tensor
from torch_geometric.nn import RotatE as BaseRotatE
from torch_geometric.nn.kge.loader import KGTripletLoader


@torch.no_grad()
def random_sample(
    model: nn.Module,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor,
    task: str,
    aux_dict: Optional[Dict] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Randomly samples negative triplets by either replacing the head or the tail (but not both),
    depending on the task specified.

    Args:
        model (nn.Module): The model being used which contains the number of nodes.
        head_index (Tensor): The head indices.
        rel_type (Tensor): The relation type indices.
        tail_index (Tensor): The tail indices.
        task (str): The task for which the model is being trained.
        aux_dict (Dict, optional): Additional auxiliary data used for node classification.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The modified head_index, rel_type, and tail_index with negative sampling applied.
    """
    num_negatives = head_index.numel() // 2
    rnd_index = torch.randint(
        model.num_nodes, head_index.size(), device=head_index.device
    )

    head_index = head_index.clone()
    tail_index = tail_index.clone()

    if task == "relation_prediction":
        selector = torch.rand(head_index.size(0)) < 0.5
        head_index[selector] = rnd_index[selector]
        tail_index[~selector] = rnd_index[~selector]
    elif task == "head_prediction":
        head_index[:num_negatives] = rnd_index[:num_negatives]
    elif task == "tail_prediction":
        tail_index[num_negatives:] = rnd_index[num_negatives:]
    elif task == "node_classification":
        if aux_dict is not None:
            aux_indices = torch.tensor(
                [int(k) for k in aux_dict.keys()],
                dtype=torch.long,
                device=tail_index.device,
            )
            rnd_aux_index = aux_indices[
                torch.randint(
                    len(aux_indices),
                    (num_negatives,),
                    device=tail_index.device,
                )
            ]
            tail_index[num_negatives:] = rnd_aux_index
        else:
            raise ValueError(
                "Non-None aux_dict is required for task 'node_classification'"
            )
    else:
        raise NotImplementedError(
            f"Task {task} not supported in random_sample"
        )

    return head_index, rel_type, tail_index


def evaluate_prediction_task(
    model: nn.Module,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor,
    batch_size: int,
    x: Optional[Tensor],
    k: Union[
        int, List[int]
    ],  # Change k to accept a single int or a list of ints
    task: str,
) -> Tuple[
    float, float, Dict[int, float]
]:  # Return hits at each k as a dictionary
    """
    Evaluates the prediction task by computing the mean rank, mean reciprocal rank (MRR), and hits at k.

    Args:
        model (nn.Module): The model to evaluate.
        head_index (Tensor): The head indices.
        rel_type (Tensor): The relation type indices.
        tail_index (Tensor): The tail indices.
        batch_size (int): The size of each batch.
        x (Tensor, optional): Additional node features.
        k (Union[int, List[int]]): The rank threshold or list of rank thresholds for calculating hits@k.
        task (str): The specific prediction task.

    Returns:
        Tuple[float, float, float]: The mean rank, MRR, and hits@k metrics.
    """
    indices = (
        torch.arange(model.num_nodes, device=tail_index.device)
        if task in ["tail_prediction", "head_prediction"]
        else torch.arange(model.num_relations, device=rel_type.device)
    )
    target_index = (
        tail_index
        if task == "tail_prediction"
        else head_index if task == "head_prediction" else rel_type
    )

    scores = compute_scores_vectorized(
        model, head_index, rel_type, tail_index, indices, task, batch_size, x
    )
    ranks = (
        scores.argsort(descending=True, dim=1) == target_index.unsqueeze(1)
    ).nonzero(as_tuple=True)[1]
    mean_rank = ranks.float().mean().item()
    mrr = (1.0 / (ranks.float() + 1)).mean().item()
    hits_at_k = {}
    if isinstance(k, int):
        k_values = [k]
    else:
        k_values = k
    for k_val in k_values:
        hits_at_k[k_val] = (ranks < k_val).float().mean().item()
    return mean_rank, mrr, hits_at_k


def compute_scores_vectorized(
    model: nn.Module,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor,
    indices: Tensor,
    task: str,
    batch_size: int,
    x: Optional[Tensor],
) -> Tensor:
    """
    Computes scores for all possible combinations of triplets in a vectorized manner.

    Args:
        model (nn.Module): The model used for scoring.
        head_index (Tensor): The head indices.
        rel_type (Tensor): The relation type indices.
        tail_index (Tensor): The tail indices.
        indices (Tensor): The indices of nodes or relations to be evaluated.
        task (str): The specific prediction task.
        batch_size (int): The size of each batch.
        x (Tensor, optional): Additional node features.

    Returns:
        Tensor: A tensor containing the scores for all combinations.
    """
    all_scores = []
    for idx_batch in indices.split(batch_size):
        if task == "tail_prediction":
            head_index_broadcasted = head_index.unsqueeze(1).repeat(
                1, idx_batch.size(0)
            )
            rel_type_broadcasted = rel_type.unsqueeze(1).repeat(
                1, idx_batch.size(0)
            )
            scores = model(
                head_index_broadcasted, rel_type_broadcasted, idx_batch, x
            )
        elif task == "head_prediction":
            rel_type_broadcasted = rel_type.unsqueeze(1).repeat(
                1, idx_batch.size(0)
            )
            tail_index_broadcasted = tail_index.unsqueeze(1).repeat(
                1, idx_batch.size(0)
            )
            scores = model(
                idx_batch, rel_type_broadcasted, tail_index_broadcasted, x
            )
        elif task == "relation_prediction":
            head_index_broadcasted = head_index.unsqueeze(1).repeat(
                1, idx_batch.size(0)
            )
            tail_index_broadcasted = tail_index.unsqueeze(1).repeat(
                1, idx_batch.size(0)
            )
            scores = model(
                head_index_broadcasted, idx_batch, tail_index_broadcasted, x
            )
        all_scores.append(scores)
    return torch.cat(all_scores, dim=1)


def evaluate_classification_task(
    model: nn.Module,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor,
    x: Optional[Tensor],
    y: Tensor,
) -> float:
    """
    Evaluates the node classification task by computing the accuracy of the predictions.

    Args:
        model (nn.Module): The model to evaluate.
        head_index (Tensor): The head indices.
        rel_type (Tensor): The relation type indices.
        tail_index (Tensor): The tail indices.
        x (Tensor, optional): Additional node features.
        y (Tensor): The ground truth labels.

    Returns:
        float: The accuracy of the classification.
    """
    if y is None or model.aux_dict is None:
        return 0.0  # Return a default value or handle the case where y or aux_dict is None

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
