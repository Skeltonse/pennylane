# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This file contains a number of attributes that may be held by operators,
and lists all operators satisfying those criteria.
"""
from inspect import isclass
from pennylane.operation import Operator, Tensor


class Attribute(set):
    r"""Class to represent a set of operators with a certain attribute.

    **Example**

    Suppose we would like to store a list of which qubit operations are
    Pauli operators. We can create a new ``Attribute``, ``pauli_ops``, like so,
    listing which operations satisfy this property.

    >>> pauli_ops = Attribute(["PauliX", "PauliZ"])

    We can check either a string or an Operation for inclusion in this set:

    >>> qml.PauliX(0) in pauli_ops
    True
    >>> "Hadamard" in pauli_ops
    False

    We can also dynamically add operators to the sets at runtime, by passing
    either a string, an operation class, or an operation itself. This is useful
    for adding custom operations to the attributes such as
    ``composable_rotations`` and ``self_inverses`` that are used in compilation
    transforms.

    >>> pauli_ops.add("PauliY")
    >>> pauli_ops
    ["PauliX", "PauliY", "PauliZ"]
    """

    def add(self, obj):
        """Add an Operator to an attribute."""
        if isinstance(obj, str):
            return super().add(obj)

        try:

            if isinstance(obj, Operator):
                return super().add(obj.name)

            if isclass(obj):
                if issubclass(obj, Operator):
                    return super().add(obj.__name__)

            raise TypeError

        except TypeError as e:
            raise TypeError(
                "Only an Operator or string representing an Operator can be added to an attribute."
            ) from e

    def __contains__(self, obj):
        """Check if the attribute contains a given Operator."""
        if isinstance(obj, str):
            return super().__contains__(obj)

        try:

            # Hotfix: return False for all tensors.
            # Can be removed or updated when tensor class is
            # improved.
            if isinstance(obj, Tensor):
                return False

            if isinstance(obj, Operator):
                return super().__contains__(obj.name)

            if isclass(obj):
                if issubclass(obj, Operator):
                    return super().__contains__(obj.__name__)

            raise TypeError

        except TypeError as e:
            raise TypeError(
                "Only an Operator or string representing an Operator can be checked for attribute inclusion."
            ) from e


composable_rotations = Attribute(
    [
        "RX",
        "RY",
        "RZ",
        "PhaseShift",
        "CRX",
        "CRY",
        "CRZ",
        "ControlledPhaseShift",
        "IsingXX",
        "IsingYY",
        "IsingZZ",
        "Rot",
    ]
)
"""Attribute: Operations for which composing multiple copies of the operation results in an
addition (or alternative accumulation) of parameters.

For example, ``qml.RZ`` is a composable rotation. Applying ``qml.RZ(0.1,
wires=0)`` followed by ``qml.RZ(0.2, wires=0)`` is equivalent to performing
a single rotation ``qml.RZ(0.3, wires=0)``.

An example for an alternative accumulation is the ``qml.Rot`` gate: although the three
angles it takes do not fulfil the composable property, the gate implements a rotation around an
axis by an effective angle which does.
"""

has_unitary_generator = Attribute(
    [
        "RX",
        "RY",
        "RZ",
        "MultiRZ",
        "PauliRot",
        "IsingXX",
        "IsingYY",
        "IsingXY",
        "IsingZZ",
        "SingleExcitationMinus",
        "SingleExcitationPlus",
        "DoubleExcitationMinus",
        "DoubleExcitationPlus",
        "OrbitalRotation",
        "FermionicSWAP",
    ]
)
"""Attribute: Operations that are generated by a unitary operator.

For example, the generator of ``qml.RZ`` is Pauli :math:`Z` with a prefactor of
:math:`-1/2`, and Pauli :math:`Z` is unitary. Contrary, the generator of
``qml.PhaseShift`` is ``np.array([[0, 0], [0, 1]])`` with a prefactor of 1,
which is not unitary. This attribute is used for decompositions in algorithms
using the Hadamard test like ``qml.metric_tensor`` when used without
approximation.
"""

self_inverses = Attribute(
    ["Hadamard", "PauliX", "PauliY", "PauliZ", "CNOT", "CZ", "CY", "CH", "SWAP", "Toffoli", "CCZ"]
)
"""Attribute: Operations that are their own inverses."""


symmetric_over_all_wires = Attribute(["CZ", "CCZ", "SWAP"])
"""Attribute: Operations that are the same if you exchange the order of wires.

For example, ``qml.CZ(wires=[0, 1])`` has the same effect as ``qml.CZ(wires=[1,
0])`` due to symmetry of the operation.
"""

symmetric_over_control_wires = Attribute(["CCZ", "Toffoli"])
"""Attribute: Controlled operations that are the same if you exchange the order of all but
the last (target) wire.

For example, ``qml.Toffoli(wires=[0, 1, 2])`` has the same effect as
``qml.Toffoli(wires=[1, 0, 2])``, but neither are the same as
``qml.Toffoli(wires=[0, 2, 1])``.
"""

diagonal_in_z_basis = Attribute(
    [
        "PauliZ",
        "S",
        "T",
        "CZ",
        "CCZ",
        "DiagonalQubitUnitary",
        "RZ",
        "PhaseShift",
        "ControlledPhaseShift",
        "MultiRZ",
        "CRZ",
        "IsingZZ",
    ]
)
"""Attribute: Operations that are diagonal in the computational basis.

For such operations, the eigenvalues provide all necessary information to
construct the matrix representation in the computational basis.

Note: Currently all gates with this attribute need
to explicitly define an eigenvalue representation.
The reason is that if this method is missing, eigenvalues are computed from the matrix
representation using ``np.linalg.eigvals``, which fails for some tensor types that the matrix
may be cast in on backpropagation devices.
"""

supports_broadcasting = Attribute(
    [
        "QubitUnitary",
        "ControlledQubitUnitary",
        "DiagonalQubitUnitary",
        "SpecialUnitary",
        "RX",
        "RY",
        "RZ",
        "PhaseShift",
        "ControlledPhaseShift",
        "Rot",
        "MultiRZ",
        "PauliRot",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        "U1",
        "U2",
        "U3",
        "IsingXX",
        "IsingYY",
        "IsingZZ",
        "IsingXY",
        "SingleExcitation",
        "SingleExcitationPlus",
        "SingleExcitationMinus",
        "DoubleExcitation",
        "DoubleExcitationPlus",
        "DoubleExcitationMinus",
        "OrbitalRotation",
        "FermionicSWAP",
        "QubitStateVector",
        "AmplitudeEmbedding",
        "AngleEmbedding",
        "IQPEmbedding",
        "QAOAEmbedding",
    ]
)
"""Attribute: Operations that support parameter broadcasting.

For such operations, the input parameters are allowed to have a single leading additional
broadcasting dimension, creating the operation with a ``batch_size`` and leading to
broadcasted tapes when used in a ``QuantumTape``.
"""
