import copy
import logging
import sqlite3
import time
from collections import namedtuple
from itertools import count

import numpy

from seba.enums import ConstraintType, OptimizationEvent

from .database import Database
from .snapshot import SebaSnapshot

OptimalResult = namedtuple(
    "OptimalResult", "batch, controls, total_objective, expected_objectives"
)


logger = logging.getLogger(__name__)


def _convert_names(control_names):
    converted_names = []
    for name in control_names:
        converted = f"{name[0]}_{name[1]}"
        if len(name) > 2:
            converted += f"-{name[2]}"
        converted_names.append(converted)
    return converted_names


class SqliteStorage:
    # This implementation builds as much as possible on the older database and
    # snapshot code, since it is meant for backwards compatibility, and should
    # not be extended with new functionality.

    def __init__(self, output_dir):
        # Optimization configuration.
        self._workflow = None

        # Internal variables.
        self._output_dir = output_dir
        self._database = Database(output_dir)
        self._control_ensemble_id = 0
        self._gradient_ensemble_id = 0
        self._simulator_results = None

    @property
    def file(self):
        return self._database.location

    def _initialize(self):
        self._database.add_experiment(
            name="optimization_experiment", start_time_stamp=time.time()
        )
        # Add configuration values.
        for control_name, initial_value, lower_bound, upper_bound in zip(
            _convert_names(self._workflow.config.variables.names),
            self._workflow.config.variables.initial_values,
            self._workflow.config.variables.lower_bounds,
            self._workflow.config.variables.upper_bounds,
        ):
            self._database.add_control_definition(
                control_name, initial_value, lower_bound, upper_bound
            )
        for name, weight, scale in zip(
            self._workflow.config.objective_functions.names,
            self._workflow.config.objective_functions.weights,
            self._workflow.config.objective_functions.scales,
        ):
            self._database.add_function(
                name=name,
                function_type="OBJECTIVE",
                weight=weight,
                normalization=1.0 / scale,
            )
        if self._workflow.config.nonlinear_constraints is not None:
            for name, scale, rhs_value, constraint_type in zip(
                self._workflow.config.nonlinear_constraints.names,
                self._workflow.config.nonlinear_constraints.scales,
                self._workflow.config.nonlinear_constraints.rhs_values,
                self._workflow.config.nonlinear_constraints.types,
            ):
                self._database.add_function(
                    name=name,
                    function_type="CONSTRAINT",
                    normalization=scale,
                    rhs_value=rhs_value,
                    constraint_type=ConstraintType(constraint_type).name.lower(),
                )
        for name, weight in zip(
            self._workflow.config.realizations.names,
            self._workflow.config.realizations.weights,
        ):
            self._database.add_realization(str(name), weight)

    def _add_batch(self, controls, perturbed_controls):
        self._database.add_batch()
        self._gradient_ensemble_id += 1
        self._control_ensemble_id = self._gradient_ensemble_id
        control_names = _convert_names(self._workflow.config.variables.names)
        for control_name, value in zip(control_names, controls):
            self._database.add_control_value(
                set_id=self._control_ensemble_id,
                control_name=control_name,
                value=value,
            )
        if perturbed_controls is not None:
            perturbed_controls = perturbed_controls.reshape(
                perturbed_controls.shape[0], -1
            )
            self._gradient_ensemble_id = self._control_ensemble_id
            for g_idx in range(perturbed_controls.shape[1]):
                self._gradient_ensemble_id += 1
                for c_idx, c_name in enumerate(control_names):
                    self._database.add_control_value(
                        set_id=self._gradient_ensemble_id,
                        control_name=c_name,
                        value=perturbed_controls[c_idx, g_idx],
                    )

    def _add_simulations(self, batch, last_result):
        self._gradient_ensemble_id = self._control_ensemble_id
        simulation_index = count()
        if last_result.functions is not None:
            for realization_name in self._workflow.config.realizations.names:
                self._database.add_simulation(
                    realization_name=str(realization_name),
                    set_id=self._control_ensemble_id,
                    sim_name=f"{batch}_{next(simulation_index)}",
                    is_gradient=False,
                )
        if last_result.gradients is not None:
            for realization_name in self._workflow.config.realizations.names:
                for _ in range(self._workflow.config.gradient.perturbation_number):
                    self._gradient_ensemble_id += 1
                    self._database.add_simulation(
                        realization_name=str(realization_name),
                        set_id=self._gradient_ensemble_id,
                        sim_name=f"{batch}_{next(simulation_index)}",
                        is_gradient=True,
                    )

    def _add_simulator_results(self, batch, objective_results, constraint_results):
        if constraint_results is None:
            results = objective_results
        else:
            results = numpy.vstack((objective_results, constraint_results))
        statuses = numpy.logical_and.reduce(numpy.isfinite(results), axis=0)
        names = self._workflow.config.objective_functions.names
        if self._workflow.config.nonlinear_constraints is not None:
            names += self._workflow.config.nonlinear_constraints.names

        for sim_idx, status in enumerate(statuses):
            sim_name = f"{batch}_{sim_idx}"
            for function_idx, name in enumerate(names):
                if status:
                    self._database.add_simulation_result(
                        sim_name, results[function_idx, sim_idx], name, 0
                    )
            self._database.set_simulation_ended(sim_name, status)
        self._database.set_batch_ended(time.time(), True)

    def _add_constraint_values(self, batch, constraint_values):
        statuses = numpy.logical_and.reduce(numpy.isfinite(constraint_values), axis=0)
        for sim_id, status in enumerate(statuses):
            if status:
                for idx, constraint_name in enumerate(
                    self._workflow.config.nonlinear_constraints.names
                ):
                    # Note the time_index=0, the database supports storing
                    # multipel time-points, but we do not support that, so we
                    # use times_index=0.
                    self._database.update_simulation_result(
                        simulation_name=f"{batch}_{sim_id}",
                        function_name=constraint_name,
                        times_index=0,
                        value=constraint_values[idx, sim_id],
                    )

    # pylint: disable=unused-argument
    def _add_gradients(self, objective_gradients):
        for grad_index, gradient in enumerate(objective_gradients):
            for control_index, control_name in enumerate(
                _convert_names(self._workflow.config.variables.names)
            ):
                self._database.add_gradient_result(
                    gradient[control_index],
                    self._workflow.config.objective_functions.names[grad_index],
                    1,
                    control_name,
                )

    def _add_total_objective(self, total_objective):
        self._database.add_calculation_result(
            set_id=self._control_ensemble_id,
            object_function_value=total_objective,
        )

    def _convert_constraints(self, constraint_results):
        constraint_results = copy.deepcopy(constraint_results)
        rhs_values = self._workflow.config.nonlinear_constraints.rhs_values
        for idx, constraint_type in enumerate(
            self._workflow.config.nonlinear_constraints.types
        ):
            constraint_results[idx] -= rhs_values[idx]
            if constraint_type == ConstraintType.LE:
                constraint_results[idx] *= -1.0
        return constraint_results

    def _handle_finished_batch_event(self, event):
        logger.debug("Storing batch results in the sqlite database")
        last_result = self._workflow.result

        last_batch = last_result.evaluations
        batch = last_result.batch

        self._add_batch(last_batch.variables, last_batch.perturbed_variables)
        self._add_simulations(batch, last_result)

        # Convert back the simulation results to the legacy format:
        objective_results = last_batch.objectives
        if last_batch.perturbed_objectives is not None:
            perturbed_objectives = last_batch.perturbed_objectives.reshape(
                last_batch.perturbed_objectives.shape[0], -1
            )
            if objective_results is None:
                objective_results = perturbed_objectives
            else:
                objective_results = numpy.hstack(
                    (last_batch.objectives, perturbed_objectives),
                )

        constraint_results = last_batch.constraints
        if self._workflow.config.nonlinear_constraints is not None:
            if last_batch.perturbed_constraints is not None:
                perturbed_constraints = last_batch.perturbed_constraints.reshape(
                    last_batch.perturbed_constraints.shape[0], -1
                )
                if constraint_results is None:
                    constraint_results = perturbed_constraints
                else:
                    constraint_results = numpy.hstack(
                        (last_batch.constraints, perturbed_constraints),
                    )
            # The legacy code converts all constraints to the form f(x) >=0:
            constraint_results = self._convert_constraints(constraint_results)

        self._add_simulator_results(batch, objective_results, constraint_results)
        if self._workflow.config.nonlinear_constraints:
            self._add_constraint_values(batch, constraint_results)
        if last_result.functions is not None:
            self._add_total_objective(last_result.functions.weighted_objective)
        if last_result.gradients is not None:
            self._add_gradients(last_result.gradients.objectives)

        merit_values = (
            self._workflow.optimizer.get_merit_values()
            if hasattr(self._workflow.optimizer, "get_merit_values")
            else []
        )
        self._database.update_calculation_result(merit_values)

        backup_data(self._database.location)

    def _handle_finished_event(self, event):
        logger.debug("Storing final results in the sqlite database")
        merit_values = (
            self._workflow.optimizer.get_merit_values()
            if hasattr(self._workflow.optimizer, "get_merit_values")
            else []
        )
        self._database.update_calculation_result(merit_values)
        self._database.set_experiment_ended(time.time())

    def handle_storage_event(self, event, workflow):
        if event == OptimizationEvent.START_OPTIMIZATION:
            if self._workflow is None:
                self._workflow = workflow
                self._initialize()
        elif event == OptimizationEvent.FINISHED_OPTIMIZATION:
            self._handle_finished_event(event)
        elif event == OptimizationEvent.FINISHED_BATCH:
            self._handle_finished_batch_event(event)

    def get_optimal_result(self):
        snapshot = SebaSnapshot(self._output_dir)
        optimum = next(
            (
                data
                for data in reversed(snapshot.get_optimization_data())
                if data.merit_flag
            ),
            None,
        )
        if optimum is None:
            return None
        objectives = snapshot.get_snapshot(batches=[optimum.batch_id])
        return OptimalResult(
            batch=optimum.batch_id,
            controls=optimum.controls,
            total_objective=optimum.objective_value,
            expected_objectives={
                name: value[0] for name, value in objectives.expected_objectives.items()
            },
        )


def backup_data(database_location: str) -> None:
    src = sqlite3.connect(database_location)
    dst = sqlite3.connect(database_location + ".backup")
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
