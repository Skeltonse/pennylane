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
Contains the classical Jacobian transform.
"""
# pylint: disable=import-outside-toplevel
import pennylane as qml
from pennylane import numpy as np


def classical_jacobian(qnode, argnum=None, expand_fn=None, trainable_only=True):
    r"""Returns a function to extract the Jacobian
    matrix of the classical part of a QNode.

    This transform allows the classical dependence between the QNode
    arguments and the quantum gate arguments to be extracted.

    Args:
        qnode (pennylane.QNode): QNode to compute the (classical) Jacobian of
        argnum (int or Sequence[int]): indices of QNode arguments with respect to which
            the (classical) Jacobian is computed
        expand_fn (None or function): an expansion function (if required) to be applied to the
            QNode quantum tape before the classical Jacobian is computed

    Returns:
        function: Function which accepts the same arguments as the QNode.
        When called, this function will return the Jacobian of the QNode
        gate arguments with respect to the QNode arguments indexed by ``argnum``.

    **Example**

    Consider the following QNode:

    >>> @qml.qnode(dev)
    ... def circuit(weights):
    ...     qml.RX(weights[0], wires=0)
    ...     qml.RY(0.2 * weights[0], wires=1)
    ...     qml.RY(2.5, wires=0)
    ...     qml.RZ(weights[1] ** 2, wires=1)
    ...     qml.RX(weights[2], wires=1)
    ...     return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    We can use this transform to extract the relationship :math:`f: \mathbb{R}^n \rightarrow
    \mathbb{R}^m` between the input QNode arguments :math:`w` and the gate arguments :math:`g`, for
    a given value of the QNode arguments:

    >>> cjac_fn = qml.transforms.classical_jacobian(circuit)
    >>> weights = np.array([1., 1., 0.6], requires_grad=True)
    >>> cjac = cjac_fn(weights)
    >>> print(cjac)
    [[1.  0.  0. ]
     [0.2 0.  0. ]
     [0.  0.  0. ]
     [0.  1.2 0. ]
     [0.  0.  1. ]]

    The returned Jacobian has rows corresponding to gate arguments, and columns
    corresponding to QNode arguments; that is,

    .. math:: J_{ij} = \frac{\partial}{\partial g_i} f(w_j).

    We can see that:

    - The zeroth element of ``weights`` is repeated on the first two gates generated by the QNode.

    - The third row consisting of all zeros indicates that the third gate ``RY(2.5)`` does not
      depend on the ``weights``.

    - The quadratic dependence of the fourth gate argument yields :math:`2\cdot 0.6=1.2`.

    .. note::

        The QNode is constructed during this operation.

    For a QNode with multiple QNode arguments, the arguments with respect to which the
    Jacobian is computed can be controlled with the ``argnum`` keyword argument.
    The output and its format depend on the backend:

    .. list-table:: Output format of ``classical_jacobian``
       :widths: 15 25 25 35
       :header-rows: 1

       * - Interface
         - ``argnum=None``
         - ``type(argnum)=int``
         - ``type(argnum) = Sequence[int]``
       * - ``'autograd'``
         - ``tuple(array)`` [1]
         - ``array``
         - ``tuple(array)``
       * - ``'jax'``
         - ``array`` [2]
         - ``array``
         - ``tuple(array)``
       * - ``'tf'``
         - ``tuple(array)``
         - ``array``
         - ``tuple(array)``
       * - ``'torch'``
         - ``tuple(array)``
         - ``array``
         - ``tuple(array)``

    [1] If there only is one trainable QNode argument, the tuple is unpacked to a
    single ``array``, as is the case for :func:`.jacobian`.

    [2] For JAX, ``argnum=None`` defaults to ``argnum=0`` in contrast to all other
    interfaces. This means that only the classical Jacobian with respect to the first
    QNode argument is computed if no ``argnum`` is provided.

    **Example with ``argnum``**

    >>> @qml.qnode(dev)
    ... def circuit(x, y, z):
    ...     qml.RX(qml.math.sin(x), wires=0)
    ...     qml.CNOT(wires=[0, 1])
    ...     qml.RY(y ** 2, wires=1)
    ...     qml.RZ(1 / z, wires=1)
    ...     return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    >>> jac_fn = qml.transforms.classical_jacobian(circuit, argnum=[1, 2])
    >>> x, y, z = np.array([0.1, -2.5, 0.71])
    >>> jac_fn(x, y, z)
    (array([-0., -5., -0.]), array([-0.        , -0.        , -1.98373339]))

    Only the Jacobians with respect to the arguments ``x`` and ``y`` were computed, and
    returned as a tuple of ``arrays``.
    """

    def classical_preprocessing(*args, **kwargs):
        """Returns the trainable gate parameters for a given QNode input."""
        kwargs.pop("shots", None)
        qnode.construct(args, kwargs)
        tape = qnode.qtape

        if expand_fn is not None:
            tape = expand_fn(tape)

        return qml.math.stack(tape.get_parameters(trainable_only=trainable_only))

    wrapper_argnum = argnum if argnum is not None else None

    def qnode_wrapper(*args, **kwargs):  # pylint: disable=inconsistent-return-statements
        old_interface = qnode.interface

        if old_interface == "auto":
            qnode.interface = qml.math.get_interface(*args, *list(kwargs.values()))

        if qnode.interface == "autograd":
            jac = qml.jacobian(classical_preprocessing, argnum=wrapper_argnum)(*args, **kwargs)

        if qnode.interface == "torch":
            import torch

            def _jacobian(*args, **kwargs):  # pylint: disable=unused-argument
                jac = torch.autograd.functional.jacobian(classical_preprocessing, args)

                torch_argnum = (
                    wrapper_argnum
                    if wrapper_argnum is not None
                    else qml.math.get_trainable_indices(args)
                )
                if np.isscalar(torch_argnum):
                    jac = jac[torch_argnum]
                else:
                    jac = tuple((jac[idx] for idx in torch_argnum))
                return jac

            jac = _jacobian(*args, **kwargs)

        if qnode.interface == "jax":
            import jax

            argnum = 0 if wrapper_argnum is None else wrapper_argnum

            def _jacobian(*args, **kwargs):
                if trainable_only:
                    _argnum = list(range(len(args)))
                    full_jac = jax.jacobian(classical_preprocessing, argnums=_argnum)(
                        *args, **kwargs
                    )
                    if np.isscalar(argnum):
                        return full_jac[argnum]

                    return tuple(full_jac[i] for i in argnum)

                return jax.jacobian(classical_preprocessing, argnums=argnum)(*args, **kwargs)

            jac = _jacobian(*args, **kwargs)

        if qnode.interface == "tf":
            import tensorflow as tf

            def _jacobian(*args, **kwargs):
                if np.isscalar(wrapper_argnum):
                    sub_args = args[wrapper_argnum]
                elif wrapper_argnum is None:
                    sub_args = args
                else:
                    sub_args = tuple((args[i] for i in wrapper_argnum))

                with tf.GradientTape() as tape:
                    gate_params = classical_preprocessing(*args, **kwargs)

                jac = tape.jacobian(gate_params, sub_args)
                return jac

            jac = _jacobian(*args, **kwargs)

        if old_interface == "auto":
            qnode.interface = "auto"

        return jac

    return qnode_wrapper
