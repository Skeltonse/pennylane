"""
A configuration class for hyperparameters that control an execution.
"""

from typing import Union, Callable

from dataclasses import dataclass


@dataclass
class ExecutionConfig:
    """All the properties required to specify how an execution occurs"""

    shots: int = None
    diff_method: Union[str, Callable] = None
    interface: str = None
