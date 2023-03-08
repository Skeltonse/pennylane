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
A module for performing QSVT with PennyLane.
"""
from .qsvt_ops import BlockEncode
from pennylane.ops.op_math import adjoint
from pennylane.ops.qubit import PCPhase


def qsvt(A, phi_vect, wires):
    """Executes the operations to perform the qsvt protocol"""
    d = len(phi_vect)
    if getattr(A, "shape", None) is None or len(A.shape) != 2:
        c = 1
        r = 1
    else:
        c, r = A.shape

    lst_operations = []

    if d % 2 == 0:
        for i in range(1, d // 2 + 1):
            lst_operations.append(BlockEncode(A, wires=wires))
            lst_operations.append(PCPhase(phi_vect[2 * i - 1], r, wires=wires))
            lst_operations.append(adjoint(BlockEncode(A, wires=wires)))
            lst_operations.append(PCPhase(phi_vect[2 * i - 2], c, wires=wires))

    else:
        for i in range(1, (d - 1) // 2 + 1):
            lst_operations.append(BlockEncode(A, wires=wires))
            lst_operations.append(PCPhase(phi_vect[2 * i], r, wires=wires))
            lst_operations.append(adjoint(BlockEncode(A, wires=wires)))
            lst_operations.append(PCPhase(phi_vect[2 * i - 1], c, wires=wires))

        lst_operations.append(BlockEncode(A, wires=wires))
        lst_operations.append(PCPhase(phi_vect[0], r, wires=wires))
