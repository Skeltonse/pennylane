# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the expval module"""
import sys

import numpy as np
import pytest

import pennylane as qml
from pennylane.measurements import Expectation


class TestExpval:
    """Tests for the expval function"""

    # TODO: Remove this when new CustomMP are the default
    def teardown_method(self):
        """Method called at the end of every test. It loops over all the calls to
        QubitDevice.sample and compares its output with the new _Sample.process method."""
        if not getattr(self, "spy", False):
            return
        if sys.version_info[1] <= 7:
            return  # skip tests for python@3.7 because call_args.kwargs is a tuple instead of a dict

        assert len(self.spy.call_args_list) > 0  # make sure method is mocked properly

        samples = self.dev._samples  # pylint: disable=protected-access
        state = self.dev._state  # pylint: disable=protected-access
        for call_args in self.spy.call_args_list:
            obs = call_args.args[1]
            shot_range, bin_size = (
                call_args.kwargs["shot_range"],
                call_args.kwargs["bin_size"],
            )
            # no need to use op, because the observable has already been applied to ``self.dev._state``
            meas = qml.expval(op=obs)
            old_res = self.dev.expval(obs, shot_range=shot_range, bin_size=bin_size)
            if self.dev.shots is None:
                new_res = meas.process_state(state=state, wires=self.dev.wires)
            else:
                new_res = meas.process_samples(
                    samples=samples, shot_range=shot_range, bin_size=bin_size
                )
            assert qml.math.allequal(old_res, new_res)

    @pytest.mark.parametrize("shots", [None, 1000, [1000, 10000]])
    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_value(self, r_dtype, mocker, shots):
        """Test that the expval interface works"""
        dev = qml.device("default.qubit", wires=2, shots=shots)
        dev.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        self.dev = circuit.device
        self.spy = mocker.spy(qml.QubitDevice, "expval")

        x = 0.54
        res = circuit(x)
        expected = -np.sin(x)

        assert np.allclose(res, expected, atol=0.05, rtol=0.05)
        assert res.dtype == r_dtype

    def test_not_an_observable(self, mocker):
        """Test that a warning is raised if the provided
        argument might not be hermitian."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.expval(qml.prod(qml.PauliX(0), qml.PauliZ(0)))

        self.dev = circuit.device
        self.spy = mocker.spy(qml.QubitDevice, "expval")

        with pytest.warns(UserWarning, match="Prod might not be hermitian."):
            _ = circuit()

    def test_observable_return_type_is_expectation(self, mocker):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Expectation`"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            res = qml.expval(qml.PauliZ(0))
            assert res.return_type is Expectation
            return res

        self.dev = circuit.device
        self.spy = mocker.spy(qml.QubitDevice, "expval")

        circuit()

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_numeric_type(self, obs):
        """Test that the numeric type is correct."""
        res = qml.expval(obs)
        assert res.numeric_type is float

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_shape(self, obs):
        """Test that the shape is correct."""
        res = qml.expval(obs)
        assert res.shape() == (1,)

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_shape_shot_vector(self, obs):
        """Test that the shape is correct with the shot vector too."""
        res = qml.expval(obs)
        shot_vector = (1, 2, 3)
        dev = qml.device("default.qubit", wires=3, shots=shot_vector)
        assert res.shape(dev) == (len(shot_vector),)

    @pytest.mark.parametrize("shots", [None, 1000, [1000, 10000]])
    def test_projector_var(self, shots, mocker):
        """Tests that the variance of a ``Projector`` object is computed correctly."""
        dev = qml.device("default.qubit", wires=3, shots=shots)

        basis_state = np.array([0, 0, 0])

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            return qml.expval(qml.Projector(basis_state, wires=range(3)))

        self.dev = circuit.device
        self.spy = mocker.spy(qml.QubitDevice, "expval")

        res = circuit()
        expected = [0.5, 0.5] if isinstance(shots, list) else 0.5

        assert np.allclose(res, expected, atol=0.02, rtol=0.02)
