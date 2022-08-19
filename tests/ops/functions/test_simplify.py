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
Unit tests for the qml.simplify function
"""
import pennylane as qml
from pennylane.tape import QuantumTape


def build_op():
    """Return function to build nested operator."""

    return qml.adjoint(
        qml.prod(
            qml.RX(1, 0) ** 1,
            qml.RY(1, 0),
            qml.prod(qml.adjoint(qml.PauliX(0)), qml.RZ(1, 0)),
            qml.RX(1, 0),
        )
    )


simplified_op = qml.prod(qml.RX(-1, 0), qml.RZ(-1, 0), qml.PauliX(0), qml.RY(-1, 0), qml.RX(-1, 0))


class TestSimplifyOperators:
    """Tests for the qml.simplify method used with operators."""

    def test_simplify_method_with_default_depth(self):
        """Test simplify method with default depth."""
        op = build_op()

        s_op = qml.simplify(op)
        assert isinstance(s_op, qml.ops.Prod)
        assert s_op.data == simplified_op.data
        assert s_op.wires == simplified_op.wires
        assert s_op.arithmetic_depth == simplified_op.arithmetic_depth

    def test_simplify_method_with_queuing(self):
        """Test the simplify method while queuing."""
        tape = QuantumTape()
        with tape:
            op = build_op()
            s_op = qml.simplify(op)
        assert len(tape.circuit) == 1
        assert tape.circuit[0] is s_op
        assert tape._queue[op]["owner"] is s_op


class TestSimplifyTapes:
    """Tests for the qml.simplify method used with tapes."""

    def test_simplify_tape(self):
        """Test the simplify method with a tape."""
        tape = QuantumTape()
        with tape:
            build_op()

        s_tape = qml.simplify(tape)
        assert len(s_tape) == 1
        s_op = s_tape[0]
        assert isinstance(s_op, qml.ops.Prod)
        assert s_op.data == simplified_op.data
        assert s_op.wires == simplified_op.wires
        assert s_op.arithmetic_depth == simplified_op.arithmetic_depth

    def test_execute_simplified_tape(self):
        """Test the execution of a simplified tape."""
        dev = qml.device("default.qubit", wires=2)
        tape = QuantumTape()
        with tape:
            _ = 3 * (5 * qml.RX(1, 0))
            qml.expval(op=qml.prod(qml.prod(qml.PauliX(0) ** 1, qml.PauliY(0)), qml.PauliZ(0)))

        simplified_tape_ops = [
            15 * qml.RX(1, 0),
            qml.expval(op=qml.prod(qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0))),
        ]
        s_tape = qml.simplify(tape)
        assert list(s_tape) == simplified_tape_ops
        assert dev.execute(tape) == dev.execute(s_tape)


class TestSimplifyQNodes:
    """Tests for the qml.simplify method used with qnodes."""

    def test_simplify_qnode(self):
        """Test the simplify method with a qnode."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def qnode():
            _ = 3 * (5 * qml.RX(1, 0))
            qml.expval(op=qml.prod(qml.prod(qml.PauliX(0) ** 1, qml.PauliY(0)), qml.PauliZ(0)))

        simplified_tape_op = 15 * qml.RX(1, 0)
        simplified_tape_obs = qml.prod(qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0))

        s_qnode = qml.simplify(qnode)
        assert s_qnode() == qnode()
        assert len(s_qnode.tape) == 1
        s_op = s_qnode.tape.operations
        s_obs = s_qnode.tape.observables
        assert isinstance(s_op, qml.ops.Prod)
        assert s_op.data == simplified_tape_op.data
        assert s_op.wires == simplified_tape_op.wires
        assert s_op.arithmetic_depth == simplified_tape_op.arithmetic_depth
        assert isinstance(s_obs, qml.ops.Prod)
        assert s_obs.data == simplified_tape_obs.data
        assert s_obs.wires == simplified_tape_obs.wires
        assert s_obs.arithmetic_depth == simplified_tape_obs.arithmetic_depth
