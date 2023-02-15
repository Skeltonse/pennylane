# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Code relevant for performing measurements on a state.
"""
from typing import Callable

from scipy.sparse import csr_matrix

from pennylane import math
from pennylane.measurements import StateMeasurement, MeasurementProcess, ExpectationMP
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .apply_operation import apply_operation


def state_diagonalizing_gates(
    measurementprocess: StateMeasurement, state: TensorLike
) -> TensorLike:
    """Apply a measurement to state when the measurement process has an observable with diagonalizing gates.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state
        state (TensorLike): state to apply the measurement to

    Returns:
        TensorLike: the result of the measurement
    """
    for op in measurementprocess.diagonalizing_gates():
        state = apply_operation(op, state)

    total_indices = len(state.shape)
    wires = Wires(range(total_indices))
    return measurementprocess.process_state(math.flatten(state), wires)


def state_hamiltonian_expval(measurementprocess: ExpectationMP, state: TensorLike) -> TensorLike:
    """Measure the expectation value of the state when the measured observable is a ``Hamiltonian`` or
    ``SparseHamiltonian``.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state
        state (TensorLike): the state to measure

    Returns:
        TensorLike: the result of the measurement
    """
    # Add case for backprop later
    total_wires = len(state.shape)
    Hmat = measurementprocess.obs.sparse_matrix(wire_order=list(range(total_wires)))
    _state = math.toarray(state).flatten()

    # Find the expectation value using the <\psi|H|\psi> matrix contraction
    bra = csr_matrix(math.conj(_state))
    ket = csr_matrix(_state[..., None])
    new_ket = csr_matrix.dot(Hmat, ket)
    res = csr_matrix.dot(bra, new_ket).toarray()[0]
    # Add broadcasting later

    return math.real(math.squeeze(res))


def get_measurement_function(
    measurementprocess: MeasurementProcess,
) -> Callable[[MeasurementProcess, TensorLike], TensorLike]:
    """Get the appropriate method for performing a measurement.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state

    Returns:
        Tensorlike: the result of the measurement
    """
    if isinstance(measurementprocess, StateMeasurement):

        if isinstance(measurementprocess, ExpectationMP) and measurementprocess.obs.name in (
            "Hamiltonian",
            "SparseHamiltonian",
        ):
            return state_hamiltonian_expval

        return state_diagonalizing_gates

    raise NotImplementedError


def measure(measurementprocess: MeasurementProcess, state: TensorLike) -> TensorLike:
    """Apply a measurement process to a state.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state
        state (TensorLike): the state to measure

    Returns:
        Tensorlike: the result of the measurement
    """
    return get_measurement_function(measurementprocess)(measurementprocess, state)
