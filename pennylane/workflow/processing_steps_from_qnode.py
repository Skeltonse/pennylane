"""
Processings steps extracted from QNode.construct
"""
from typing import Sequence

import pennylane as qml

from pennylane.tape import QuantumScript

from .execution_config import ExecutionConfig

QScriptBatch = Sequence[QuantumScript]


class PreprocessingError(Exception):
    """Generic exception for something not being supported during preprocessing."""


def set_trainable_params(
    batch_qscript: QScriptBatch, execution_config: ExecutionConfig = None
) -> tuple[QScriptBatch, ExecutionConfig]:
    """Sets the trainable indices for each quantum script in the batch.

    SIDE EFFECT: This sets the ``trainable_params`` property on the input scripts
    """

    def _single_set_trainable_params(qscript: QuantumScript) -> QuantumScript:
        params = qscript.get_parameters(trainable_only=False)
        qscript.trainable_params = qml.math.get_trainable_indices(params)
        return qscript

    new_batch = [_single_set_trainable_params(qs) for qs in batch_qscript]
    return new_batch, execution_config


def validate_SparseHamiltonian_backprop(
    batch_qscript: QScriptBatch, execution_config: ExecutionConfig = None
) -> tuple[QScriptBatch, ExecutionConfig]:
    """Verifies that no measurements of sparse hamiltonians occur when in backprop mode."""

    def _check_no_SparseHamiltonian(qscript: QuantumScript) -> None:
        for m in qscript.measurements:
            if isinstance(m.obs, qml.SparseHamiltonian):
                raise PreprocessingError("SparseHamiltonian not supported with backprop")

    if execution_config.diff_method == "backprop":
        _ = [_check_no_SparseHamiltonian(qs) for qs in batch_qscript]
    return batch_qscript, execution_config


def defer_measurements(
    batch_qscript: QScriptBatch, execution_config: ExecutionConfig = None
) -> tuple[QScriptBatch, ExecutionConfig]:
    """Applies qml.defer_meausrements if any midmeasurements are present."""

    def _single_defer_measurements(qscript: QuantumScript) -> QuantumScript:
        def is_mid_measure(obj):
            return getattr(obj, "return_type", None) == qml.measurements.MidMeasure

        if any(is_mid_measure(op) for op in qscript.operations):
            return qml.defer_measurements(qscript)
        return qscript

    return [_single_defer_measurements(qs) for qs in batch_qscript], execution_config


# pylint: disable=protected-access
def validate_qfunc_output_matches_measurements(
    batch_qscript: QScriptBatch, execution_config: ExecutionConfig = None
) -> tuple[QScriptBatch, ExecutionConfig]:
    """Validates the qfunc output property with respect to the qscript measurements."""

    def _validate_qfunc_output_matches_meausrements(qscript: QuantumScript) -> QuantumScript:
        if getattr(qscript, "_qfunc_output", None) is None:
            return qscript

        if isinstance(qscript._qfunc_output, qml.numpy.ndarray):  # would this ever actually happen?
            measurement_processes = tuple(qscript.measurements)
        elif not isinstance(qscript._qfunc_output, Sequence):
            measurement_processes = (qscript._qfunc_output,)
        else:
            measurement_processes = qscript._qfunc_output

        if any(
            not isinstance(m, qml.measurements.MeasurementProcess) for m in measurement_processes
        ):
            raise PreprocessingError(
                "A quantum function must return either a single measurement, "
                "or a nonempty sequence of measurements."
            )
        if any(ret != m for ret, m in zip(measurement_processes, qscript.measurements)):
            raise PreprocessingError(
                "All measurements must be returned in the order they are measured."
            )
        return qscript

    return [
        _validate_qfunc_output_matches_meausrements(qs) for qs in batch_qscript
    ], execution_config
