"""MODULE DOCSTRING


"""

from .processing_steps_from_qnode import (
    set_trainable_params,
    validate_SparseHamiltonian_backprop,
    defer_measurements,
    validate_qfunc_output_matches_measurements,
)
from .execution_config import ExecutionConfig
