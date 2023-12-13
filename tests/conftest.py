from dataclasses import dataclass
from functools import partial
from itertools import count
from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from seba.evaluator import Evaluator, EvaluatorMetaData
from seba.utils.scaling import scale_back_variables

_Function = Callable[[NDArray[np.float64], Any], float]


@dataclass
class FunctionContext:
    realization: int
    index: int


def _function_runner(
    variables: NDArray[np.float64],
    metadata: EvaluatorMetaData,
    functions: List[_Function],
    counter: Iterator[int],
) -> Tuple[int, NDArray[np.float64], Optional[NDArray[np.float64]]]:
    unscaled_variables = scale_back_variables(metadata.config, variables, axis=-1)
    if unscaled_variables is not None:
        variables = unscaled_variables
    objective_count = metadata.config.objective_functions.weights.size
    constraint_count = (
        0
        if metadata.config.nonlinear_constraints is None
        else metadata.config.nonlinear_constraints.rhs_values.size
    )
    objective_results = np.zeros(
        (variables.shape[0], objective_count),
        dtype=np.float64,
    )
    constraint_results = (
        np.zeros((variables.shape[0], constraint_count), dtype=np.float64)
        if constraint_count > 0
        else None
    )
    for sim, realization in enumerate(metadata.realizations):
        context = FunctionContext(realization=realization, index=sim)
        for idx in range(objective_count):
            if (
                metadata.active_objectives is None
                or metadata.active_objectives[idx, sim]
            ):
                function = functions[idx]
                objective_results[sim, idx] = function(variables[sim, :], context)
        for idx in range(constraint_count):
            if (
                metadata.active_constraints is None
                or metadata.active_constraints[idx, sim]
            ):
                function = functions[idx + objective_count]
                assert constraint_results is not None
                constraint_results[sim, idx] = function(variables[sim, :], context)
    return next(counter), objective_results, constraint_results


def _compute_distance_squared(
    variables: NDArray[np.float64],
    context: Any,  # noqa: ARG001
    target: NDArray[np.float64],
) -> float:
    return float(((variables - target) ** 2).sum())


@pytest.fixture(name="test_functions", scope="session")
def fixture_test_functions() -> Tuple[_Function, _Function]:
    return (
        partial(_compute_distance_squared, target=[0.5, 0.5, 0.5]),
        partial(_compute_distance_squared, target=[-1.5, -1.5, 0.5]),
    )


@pytest.fixture(scope="session")
def evaluator(test_functions: Any) -> Callable[[List[_Function]], Evaluator]:
    def _evaluator(test_functions: List[_Function] = test_functions) -> Evaluator:
        return partial(_function_runner, functions=test_functions, counter=count())

    return _evaluator