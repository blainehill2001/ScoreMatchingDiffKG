from typing import Dict
from typing import Tuple
from typing import Union

KG_Completion_Metrics = Union[
    Tuple[float, float, Dict[int, float]],
    Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        Dict[int, float],
        Dict[int, float],
        Dict[int, float],
    ],
]
