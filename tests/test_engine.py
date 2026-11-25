"""Comprehensive pytest suite for SEIZ schemas and ODE simulation engine."""

from __future__ import annotations

import ast
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import structlog
from prometheus_client import Counter, Histogram, Summary
from pydantic import ValidationError
from scipy.integrate._ivp.ivp import OdeResult
from scipy.optimize import OptimizeResult, OptimizeWarning

import src.simulation.engine as engine_module
from src.cpp_core import seiz_cpp
from src.models.schemas import InitialState, SEIZParameters
from src.simulation.calibration import (
    OPTIMIZATION_FAILURES_TOTAL,
    OptimizationConvergenceError,
    ParameterEstimator,
)
from src.simulation.engine import (
    CPP_INTEGRATION_DURATION_MS,
    CPP_INTEGRATION_SUMMARY_MS,
    HAS_CPP_CORE,
    SEIZSimulator,
    SimulationResult,
    configure_structlog,
)


class TestInitialStateSchema:
    """Validation and helper tests for InitialState."""

    def test_valid_initial_state_and_aliases(self) -> None:
        state_short = InitialState(S0=900, E0=50, I0=30, Z0=20)
        state_underscore = InitialState(S_0=900, E_0=50, I_0=30, Z_0=20)

        assert state_short == state_underscore
        assert state_short.N == 1000
        assert (
            state_short.S_0,
            state_short.E_0,
            state_short.I_0,
            state_short.Z_0,
        ) == (900, 50, 30, 20)
        np.testing.assert_array_equal(state_short.to_array(), np.array([900.0, 50.0, 30.0, 20.0]))

    def test_zero_in_some_compartments_allowed(self) -> None:
        state = InitialState(S0=100, E0=0, I0=0, Z0=0)
        assert state.N == 100

    def test_all_zero_population_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InitialState(S0=0, E0=0, I0=0, Z0=0)

    @pytest.mark.parametrize("field", ["S0", "E0", "I0", "Z0"])
    @pytest.mark.parametrize("bad_value", [-1, -100, 10.5, 100.0, True, False, "10"])
    def test_rejects_negative_float_bool_and_str(self, field: str, bad_value: object) -> None:
        payload: dict[str, object] = {"S0": 100, "E0": 10, "I0": 5, "Z0": 5}
        payload[field] = bad_value
        with pytest.raises(ValidationError):
            InitialState(**payload)

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            InitialState(S0=100, E0=10, I0=5, Z0=5, extra_field=1)


class TestSEIZParametersSchema:
    """Validation tests for SEIZParameters."""

    def test_valid_parameters_including_boundaries(self) -> None:
        params = SEIZParameters(beta=0.0, p=0.0, l=1.0, b=0.0, rho=0.0, epsilon=0.0)
        assert params.p == 0.0
        assert params.l == 1.0

    @pytest.mark.parametrize("prob_field", ["p", "l"])
    @pytest.mark.parametrize(
        "bad_prob",
        [-0.01, -1.0, 1.0001, 2.5, True, False, float("nan"), float("inf")],
    )
    def test_probability_bounds_enforced(self, prob_field: str, bad_prob: float | bool) -> None:
        payload: dict[str, object] = {
            "beta": 0.4,
            "p": 0.5,
            "l": 0.5,
            "b": 0.2,
            "rho": 0.1,
            "epsilon": 0.05,
        }
        payload[prob_field] = bad_prob
        with pytest.raises(ValidationError):
            SEIZParameters(**payload)

    @pytest.mark.parametrize("rate_field", ["beta", "b", "rho", "epsilon"])
    @pytest.mark.parametrize(
        "bad_rate",
        [-1e-6, -5.0, True, float("nan"), float("inf"), float("-inf")],
    )
    def test_rate_non_negativity_and_finiteness_enforced(
        self,
        rate_field: str,
        bad_rate: float | bool,
    ) -> None:
        payload: dict[str, object] = {
            "beta": 0.4,
            "p": 0.5,
            "l": 0.5,
            "b": 0.2,
            "rho": 0.1,
            "epsilon": 0.05,
        }
        payload[rate_field] = bad_rate
        with pytest.raises(ValidationError):
            SEIZParameters(**payload)


class TestSEIZDerivatives:
    """Unit tests for SEIZSimulator._seiz_derivatives."""

    def test_exact_derivatives_and_zero_sum_invariant(self) -> None:
        params = SEIZParameters(beta=0.5, p=0.4, l=0.6, b=0.2, rho=0.3, epsilon=0.1)
        y = np.array([500.0, 200.0, 200.0, 100.0], dtype=np.float64)
        # N = 1000
        # si_flow = 0.5 * 500 * 200 / 1000 = 50.0
        # sz_flow = 0.2 * 500 * 100 / 1000 = 10.0
        # ei_flow = 0.3 * 200 * 200 / 1000 = 12.0
        # inc_flow = 0.1 * 200 = 20.0
        # dS = -50.0 - 10.0 = -60.0
        # dE = 0.6 * 50.0 + 0.4 * 10.0 - 12.0 - 20.0 = 30.0 + 4.0 - 32.0 = 2.0
        # dI = 0.4 * 50.0 + 12.0 + 20.0 = 20.0 + 32.0 = 52.0
        # dZ = 0.6 * 10.0 = 6.0
        expected = np.array([-60.0, 2.0, 52.0, 6.0], dtype=np.float64)

        dy = SEIZSimulator._seiz_derivatives(0.0, y, params)
        np.testing.assert_allclose(dy, expected, rtol=1e-12, atol=1e-12)
        assert np.isclose(np.sum(dy), 0.0, atol=1e-12)

    def test_derivatives_reject_invalid_state_shape_or_nonpositive_n(self) -> None:
        params = SEIZParameters(beta=0.5, p=0.4, l=0.6, b=0.2, rho=0.3, epsilon=0.1)
        with pytest.raises(ValueError):
            SEIZSimulator._seiz_derivatives(0.0, np.array([100.0, 0.0, 0.0]), params)
        with pytest.raises(ValueError):
            SEIZSimulator._seiz_derivatives(0.0, np.array([0.0, 0.0, 0.0, 0.0]), params)


class TestSEIZSimulatorMassConservationAndInvariants:
    """Integration tests validating mass conservation (tol=1e-6), stiffness, and invariants."""

    def test_mass_conservation_standard_epidemic(self) -> None:
        simulator = SEIZSimulator()
        state = InitialState(S0=9900, E0=50, I0=30, Z0=20)
        params = SEIZParameters(beta=0.6, p=0.25, l=0.4, b=0.3, rho=0.2, epsilon=0.1)

        result = simulator.solve(state, params, t_span=(0.0, 100.0))

        assert isinstance(result, SimulationResult)
        assert isinstance(result, OdeResult)
        assert isinstance(result, dict)
        assert result.success is True
        assert result["success"] is True

        totals = result["S"] + result["E"] + result["I"] + result["Z"]
        assert totals.shape == result["t"].shape
        np.testing.assert_allclose(totals, float(state.N), rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(result.N, float(state.N), rtol=0.0, atol=1e-6)

        # Non-negativity and monotonicity invariants
        assert np.all(result.y >= -1e-9)
        assert np.all(np.diff(result.S) <= 1e-9)
        assert np.all(np.diff(result.Z) >= -1e-9)

    @pytest.mark.parametrize(
        "params_dict",
        [
            {"beta": 500.0, "p": 0.05, "l": 0.2, "b": 100.0, "rho": 250.0, "epsilon": 1e-4},
            {"beta": 1e4, "p": 0.01, "l": 0.8, "b": 500.0, "rho": 1000.0, "epsilon": 1e-3},
        ],
    )
    def test_mass_conservation_stiff_ode_lsoda(self, params_dict: dict[str, float]) -> None:
        simulator = SEIZSimulator()
        state = InitialState(S0=10_000, E0=100, I0=10, Z0=5)
        params = SEIZParameters(**params_dict)

        result = simulator.solve(state, params, t_span=(0.0, 50.0), num_points=400)

        totals = result.S + result.E + result.I + result.Z
        np.testing.assert_allclose(totals, float(state.N), rtol=0.0, atol=1e-6)
        assert np.all(result.y >= -1e-7)
        assert np.all(np.diff(result.S) <= 1e-7)
        assert np.all(np.diff(result.Z) >= -1e-7)

    @pytest.mark.parametrize("p_val,l_val", [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)])
    def test_mass_conservation_boundary_probabilities(self, p_val: float, l_val: float) -> None:
        simulator = SEIZSimulator()
        state = InitialState(S0=900, E0=50, I0=25, Z0=25)
        params = SEIZParameters(beta=0.5, p=p_val, l=l_val, b=0.4, rho=0.25, epsilon=0.1)

        result = simulator.solve(state, params, t_span=(0.0, 80.0))

        totals = result.S + result.E + result.I + result.Z
        np.testing.assert_allclose(totals, float(state.N), rtol=0.0, atol=1e-6)
        assert np.all(result.y >= -1e-9)

    def test_zero_rates_preserve_constant_state(self) -> None:
        simulator = SEIZSimulator()
        state = InitialState(S0=700, E0=150, I0=100, Z0=50)
        params = SEIZParameters(beta=0.0, p=0.5, l=0.5, b=0.0, rho=0.0, epsilon=0.0)

        result = simulator.solve(state, params, t_span=(0.0, 60.0), num_points=100)

        for idx, expected_val in enumerate([700.0, 150.0, 100.0, 50.0]):
            np.testing.assert_allclose(result.y[idx], expected_val, rtol=0.0, atol=1e-6)
        total_pop = result.S + result.E + result.I + result.Z
        np.testing.assert_allclose(total_pop, float(state.N), rtol=0.0, atol=1e-6)

    @pytest.mark.parametrize("scale", [1, 100, 10_000, 100_000])
    def test_mass_conservation_and_scale_invariance_across_populations(self, scale: int) -> None:
        simulator = SEIZSimulator()
        base_state = InitialState(S0=80, E0=10, I0=5, Z0=5)
        scaled_state = InitialState(S0=80 * scale, E0=10 * scale, I0=5 * scale, Z0=5 * scale)
        params = SEIZParameters(beta=0.45, p=0.3, l=0.7, b=0.25, rho=0.15, epsilon=0.05)

        t_grid = np.linspace(0.0, 40.0, 81)
        base_res = simulator.solve(base_state, params, t_span=t_grid)
        scaled_res = simulator.solve(scaled_state, params, t_span=(0.0, 40.0), t_eval=t_grid)

        totals = scaled_res.S + scaled_res.E + scaled_res.I + scaled_res.Z
        np.testing.assert_allclose(totals, float(scaled_state.N), rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(
            scaled_res.y / scaled_state.N,
            base_res.y / base_state.N,
            rtol=1e-6,
            atol=1e-8,
        )

    def test_zero_length_span_and_invalid_solver_arguments(self) -> None:
        simulator = SEIZSimulator()
        state = InitialState(S0=100, E0=0, I0=1, Z0=0)
        params = SEIZParameters(beta=0.3, p=0.5, l=0.5, b=0.1, rho=0.1, epsilon=0.05)

        res_zero = simulator.solve(state, params, t_span=(5.0, 5.0))
        assert res_zero.success is True
        assert res_zero.t.tolist() == [5.0]
        np.testing.assert_allclose(res_zero.y[:, 0], state.to_array(), rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(res_zero.N, float(state.N), rtol=0.0, atol=1e-6)

        res_from_dicts = simulator.solve(
            {"S0": 100, "E0": 0, "I0": 1, "Z0": 0},
            {"beta": 0.3, "p": 0.5, "l": 0.5, "b": 0.1, "rho": 0.1, "epsilon": 0.05},
            t_span=(0.0, 10.0),
            t_eval=[0.0, 5.0, 10.0],
        )
        assert res_from_dicts.t.tolist() == [0.0, 5.0, 10.0]
        np.testing.assert_allclose(res_from_dicts.N, 101.0, rtol=0.0, atol=1e-6)

        with pytest.raises(ValueError):
            simulator.solve(state, params, t_span=(10.0, 0.0))
        with pytest.raises(ValueError):
            simulator.solve(state, params, t_span=(0.0, 10.0), num_points=1)
        with pytest.raises(ValueError):
            simulator.solve(state, params, t_span=(0.0, 10.0), t_eval=[0.0, 12.0])
        with pytest.raises(ValueError):
            simulator.solve(state, params, t_span=(0.0, 10.0), t_eval=[0.0, 5.0, 5.0])
        with pytest.raises(ValueError):
            simulator.solve(state, params, t_span=(0.0, 10.0), rtol=0.0)
        with pytest.raises(ValueError):
            simulator.solve(state, params, t_span=(0.0, 10.0), atol=-1e-10)
        with pytest.raises(TypeError):
            simulator.solve("invalid", params, t_span=(0.0, 10.0))  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            simulator.solve(state, "invalid", t_span=(0.0, 10.0))  # type: ignore[arg-type]


class TestParameterEstimator:
    """Unit and integration tests for SEIZ inverse-problem calibration."""

    @staticmethod
    def _generate_synthetic_data() -> tuple[
        InitialState,
        SEIZParameters,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        state = InitialState(S0=950, E0=25, I0=15, Z0=10)
        true_params = SEIZParameters(
            beta=0.65,
            p=0.35,
            l=0.45,
            b=0.25,
            rho=0.2,
            epsilon=0.08,
        )
        t_obs = np.linspace(0.0, 40.0, 25)
        sim = SEIZSimulator().solve(state, true_params, t_span=(0.0, 40.0), t_eval=t_obs)
        return state, true_params, t_obs, sim.I, sim.Z

    def test_objective_function_zero_at_truth_and_matches_manual_sse(self) -> None:
        state, true_params, t_obs, i_obs, z_obs = self._generate_synthetic_data()
        estimator = ParameterEstimator()

        true_vec = np.array(
            [
                true_params.beta,
                true_params.p,
                true_params.l,
                true_params.b,
                true_params.rho,
                true_params.epsilon,
            ]
        )
        sse_truth = estimator._objective_function(true_vec, t_obs, i_obs, z_obs, state)
        assert np.isclose(sse_truth, 0.0, atol=1e-10)

        perturbed_params = SEIZParameters(
            beta=0.5,
            p=0.2,
            l=0.6,
            b=0.35,
            rho=0.1,
            epsilon=0.12,
        )
        pert_vec = np.array(
            [
                perturbed_params.beta,
                perturbed_params.p,
                perturbed_params.l,
                perturbed_params.b,
                perturbed_params.rho,
                perturbed_params.epsilon,
            ]
        )
        sim_pert = estimator.simulator.solve(
            state, perturbed_params, t_span=(float(t_obs[0]), float(t_obs[-1])), t_eval=t_obs
        )
        expected_sse = float(np.sum((sim_pert.I - i_obs) ** 2) + np.sum((sim_pert.Z - z_obs) ** 2))

        sse_pert = estimator._objective_function(pert_vec, t_obs, i_obs, z_obs, state)
        assert sse_pert > 0.0
        assert np.isclose(sse_pert, expected_sse, rtol=1e-10, atol=1e-10)

    def test_fit_to_data_recovers_trajectory_with_injected_and_default_guess(self) -> None:
        state, _, t_obs, i_obs, z_obs = self._generate_synthetic_data()
        estimator = ParameterEstimator()

        initial_guess = [0.4, 0.5, 0.3, 0.15, 0.1, 0.05]
        fitted = estimator.fit_to_data(
            t_observed=t_obs,
            I_observed=i_obs,
            Z_observed=z_obs,
            initial_state=state,
            initial_guess=initial_guess,
        )

        assert isinstance(fitted, SEIZParameters)
        assert 0.0 <= fitted.p <= 1.0
        assert 0.0 <= fitted.l <= 1.0
        assert fitted.beta >= 0.0
        assert fitted.b >= 0.0
        assert fitted.rho >= 0.0
        assert fitted.epsilon >= 0.0

        fitted_sim = estimator.simulator.solve(state, fitted, t_span=(0.0, 40.0), t_eval=t_obs)
        rmse = np.sqrt(
            (np.mean((fitted_sim.I - i_obs) ** 2) + np.mean((fitted_sim.Z - z_obs) ** 2)) / 2.0
        )
        assert rmse < 1.0

        fitted_default = estimator.fit_to_data(
            t_observed=t_obs,
            I_observed=i_obs,
            Z_observed=z_obs,
            initial_state=state,
            initial_guess=None,
        )
        assert isinstance(fitted_default, SEIZParameters)
        default_sim = estimator.simulator.solve(
            state, fitted_default, t_span=(0.0, 40.0), t_eval=t_obs
        )
        default_rmse = np.sqrt(
            (np.mean((default_sim.I - i_obs) ** 2) + np.mean((default_sim.Z - z_obs) ** 2)) / 2.0
        )
        assert default_rmse < 1.0

    def test_graceful_degradation_on_convergence_warnings(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state, _, t_obs, i_obs, z_obs = self._generate_synthetic_data()
        estimator = ParameterEstimator()

        # Early termination (maxiter=1) returns best-fit parameters cleanly without raising.
        early_fitted = estimator.fit_to_data(
            t_observed=t_obs,
            I_observed=i_obs,
            Z_observed=z_obs,
            initial_state=state,
            initial_guess=[0.4, 0.5, 0.3, 0.15, 0.1, 0.05],
            maxiter=1,
        )
        assert isinstance(early_fitted, SEIZParameters)

        # Explicit OptimizeWarning from scipy.optimize.minimize is caught and degraded gracefully.
        def fake_minimize_with_warning(*args: object, **kwargs: object) -> OptimizeResult:
            del args, kwargs
            warnings.warn(
                "L-BFGS-B did not converge to tolerance.",
                OptimizeWarning,
                stacklevel=2,
            )
            return OptimizeResult(
                x=np.array([0.55, 0.4, 0.4, 0.25, 0.18, 0.09], dtype=np.float64),
                fun=12.5,
                success=False,
                status=1,
                message="STOP: TOTAL NO. of ITERATIONS REACHED LIMIT",
                hess_inv=np.eye(6, dtype=np.float64),
            )

        monkeypatch.setattr("src.simulation.calibration.minimize", fake_minimize_with_warning)
        with warnings.catch_warnings(record=True) as leaked:
            warnings.simplefilter("always")
            degraded = estimator.fit_to_data(
                t_observed=t_obs,
                I_observed=i_obs,
                Z_observed=z_obs,
                initial_state=state,
            )

        assert len(leaked) == 0
        assert isinstance(degraded, SEIZParameters)
        assert np.isclose(degraded.beta, 0.55)

    def test_raises_optimization_convergence_error_on_matrix_inversion_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state, _, t_obs, i_obs, z_obs = self._generate_synthetic_data()
        estimator = ParameterEstimator()

        # Case 1: Singular inverse Hessian matrix (zeros -> np.linalg.inv raises LinAlgError).
        def fake_minimize_singular_hess(*args: object, **kwargs: object) -> OptimizeResult:
            del args, kwargs
            return OptimizeResult(
                x=np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.1], dtype=np.float64),
                fun=10.0,
                success=False,
                hess_inv=np.zeros((6, 6), dtype=np.float64),
            )

        monkeypatch.setattr("src.simulation.calibration.minimize", fake_minimize_singular_hess)
        with pytest.raises(OptimizationConvergenceError):
            estimator.fit_to_data(t_obs, i_obs, z_obs, state)

        # Case 2: minimize itself raises LinAlgError during matrix inversion.
        def fake_minimize_linalg_error(*args: object, **kwargs: object) -> OptimizeResult:
            del args, kwargs
            raise np.linalg.LinAlgError("Singular matrix during BFGS update")

        monkeypatch.setattr("src.simulation.calibration.minimize", fake_minimize_linalg_error)
        with pytest.raises(OptimizationConvergenceError):
            estimator.fit_to_data(t_obs, i_obs, z_obs, state)

        # Case 3: Non-finite solution vector returned.
        def fake_minimize_nan_solution(*args: object, **kwargs: object) -> OptimizeResult:
            del args, kwargs
            return OptimizeResult(
                x=np.array([np.nan, 0.5, 0.5, 0.2, 0.2, 0.1], dtype=np.float64),
                fun=10.0,
                success=False,
                hess_inv=np.eye(6, dtype=np.float64),
            )

        monkeypatch.setattr("src.simulation.calibration.minimize", fake_minimize_nan_solution)
        with pytest.raises(OptimizationConvergenceError):
            estimator.fit_to_data(t_obs, i_obs, z_obs, state)


class TestCppCoreAndBenchmark:
    """Tests for compiled C++ pybind11 RK4 module, SciPy fallback, and 1,000-sim benchmark."""

    def test_cpp_module_derivatives_and_rk4_accuracy(self) -> None:
        assert HAS_CPP_CORE is True

        state = InitialState(S0=9900, E0=50, I0=30, Z0=20)
        params = SEIZParameters(beta=0.6, p=0.25, l=0.4, b=0.3, rho=0.2, epsilon=0.1)
        y0 = [float(state.S0), float(state.E0), float(state.I0), float(state.Z0)]
        p_vec = [params.beta, params.p, params.l, params.b, params.rho, params.epsilon]

        cpp_deriv = np.asarray(seiz_cpp.seiz_derivatives(y0, p_vec), dtype=np.float64)
        py_deriv = SEIZSimulator._seiz_derivatives(0.0, np.asarray(y0), params)
        np.testing.assert_allclose(cpp_deriv, py_deriv, rtol=1e-12, atol=1e-12)

        cpp_dict = seiz_cpp.solve_rk4(y0, p_vec, 0.2, 500, 0.0)
        cpp_scalar = seiz_cpp.solve_rk4(*y0, *p_vec, 0.2, 500, 0.0)
        cpp_flat = seiz_cpp.solve_rk4_flat(y0, p_vec, 0.2, 500)

        assert cpp_dict["t"].shape == (501,)
        assert cpp_dict["y_flat"].shape == (4 * 501,)
        np.testing.assert_allclose(cpp_dict["N"], float(state.N), rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(cpp_dict["y_flat"], cpp_scalar["y_flat"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cpp_dict["y_flat"], cpp_flat, rtol=1e-12, atol=1e-12)

        cpp_sim = SEIZSimulator(use_cpp=True).solve(
            state, params, t_span=(0.0, 100.0), num_points=501
        )
        scipy_sim = SEIZSimulator(use_cpp=False).solve(
            state, params, t_span=(0.0, 100.0), num_points=501
        )
        assert "C++ RK4 core" in cpp_sim.message
        assert "C++ RK4 core" not in scipy_sim.message
        np.testing.assert_allclose(cpp_sim.y, scipy_sim.y, rtol=1e-4, atol=1e-2)

        # Non-uniform and offset t_eval handling in C++ wrapper
        non_uniform_t = [2.5, 5.0, 15.0, 40.0]
        cpp_non_uni = SEIZSimulator(use_cpp=True).solve(
            state, params, t_span=(0.0, 50.0), t_eval=non_uniform_t
        )
        scipy_non_uni = SEIZSimulator(use_cpp=False).solve(
            state, params, t_span=(0.0, 50.0), t_eval=non_uniform_t
        )
        np.testing.assert_allclose(cpp_non_uni.N, float(state.N), rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(cpp_non_uni.y, scipy_non_uni.y, rtol=1e-4, atol=5e-2)

        cpp_single = SEIZSimulator(use_cpp=True).solve(
            state, params, t_span=(0.0, 10.0), t_eval=[5.0]
        )
        scipy_single = SEIZSimulator(use_cpp=False).solve(
            state, params, t_span=(0.0, 10.0), t_eval=[5.0]
        )
        np.testing.assert_allclose(cpp_single.y, scipy_single.y, rtol=1e-4, atol=5e-2)

    def test_scipy_fallback_when_cpp_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = InitialState(S0=1000, E0=20, I0=10, Z0=5)
        params = SEIZParameters(beta=0.5, p=0.3, l=0.5, b=0.25, rho=0.15, epsilon=0.05)

        monkeypatch.setattr(engine_module, "_cpp_engine", None)
        monkeypatch.setattr(engine_module, "HAS_CPP_CORE", False)

        fallback_sim = SEIZSimulator().solve(state, params, t_span=(0.0, 50.0), num_points=200)
        assert fallback_sim.success is True
        assert "C++ RK4 core" not in fallback_sim.message
        np.testing.assert_allclose(fallback_sim.N, float(state.N), rtol=0.0, atol=1e-6)

    def test_benchmark_1000_simulations_cpp_vs_scipy(self) -> None:
        state = InitialState(S0=9900, E0=50, I0=30, Z0=20)
        params = SEIZParameters(beta=0.6, p=0.25, l=0.4, b=0.3, rho=0.2, epsilon=0.1)

        cpp_simulator = SEIZSimulator(use_cpp=True)
        scipy_simulator = SEIZSimulator(use_cpp=False)

        t_cpp_start = time.perf_counter()
        last_cpp = None
        for _ in range(1000):
            last_cpp = cpp_simulator.solve(state, params, t_span=(0.0, 100.0), num_points=500)
        cpp_elapsed = time.perf_counter() - t_cpp_start

        t_scipy_start = time.perf_counter()
        last_scipy = None
        for _ in range(1000):
            last_scipy = scipy_simulator.solve(state, params, t_span=(0.0, 100.0), num_points=500)
        scipy_elapsed = time.perf_counter() - t_scipy_start

        assert last_cpp is not None and last_scipy is not None
        np.testing.assert_allclose(last_cpp.N, float(state.N), rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(last_scipy.N, float(state.N), rtol=0.0, atol=1e-6)

        structlog.get_logger(__name__).info(
            "benchmark_1000_simulations",
            cpp_rk4_seconds=round(cpp_elapsed, 4),
            scipy_lsoda_seconds=round(scipy_elapsed, 4),
            speedup_factor=round(scipy_elapsed / cpp_elapsed, 2),
        )
        assert cpp_elapsed < scipy_elapsed


class TestGillespieStochasticSolver:
    """Tests for multi-threaded C++ Gillespie algorithm and percentile aggregation."""

    def test_cpp_gillespie_percentiles_and_conservation(self) -> None:
        state = InitialState(S0=950, E0=25, I0=15, Z0=10)
        params = SEIZParameters(beta=0.6, p=0.25, l=0.4, b=0.3, rho=0.2, epsilon=0.1)
        y0 = [float(state.S0), float(state.E0), float(state.I0), float(state.Z0)]
        p_vec = [params.beta, params.p, params.l, params.b, params.rho, params.epsilon]

        cpp_res = seiz_cpp.solve_gillespie(y0, p_vec, 0.25, 200, 0.0, 1000, 42)
        p5_t, p50_t, p95_t = seiz_cpp.solve_gillespie_percentiles(
            y0, p_vec, 0.25, 200, 0.0, 1000, 42
        )

        assert cpp_res["t"].shape == (201,)
        assert cpp_res["p5"].shape == (4 * 201,)
        assert cpp_res["p50"].shape == (4 * 201,)
        assert cpp_res["p95"].shape == (4 * 201,)
        np.testing.assert_allclose(cpp_res["p5"], p5_t)
        np.testing.assert_allclose(cpp_res["p50"], p50_t)
        np.testing.assert_allclose(cpp_res["p95"], p95_t)
        np.testing.assert_allclose(cpp_res["N"], float(state.N), rtol=0.0, atol=1e-9)

        p5 = cpp_res["p5"].reshape(4, 201)
        p50 = cpp_res["p50"].reshape(4, 201)
        p95 = cpp_res["p95"].reshape(4, 201)
        assert np.all(p5 >= 0.0)
        assert np.all(p5 <= p50 + 1e-12)
        assert np.all(p50 <= p95 + 1e-12)
        assert np.all(p95 <= float(state.N))

    def test_simulator_solve_stochastic_and_python_fallback(self) -> None:
        state = InitialState(S0=950, E0=25, I0=15, Z0=10)
        params = SEIZParameters(beta=0.65, p=0.3, l=0.4, b=0.25, rho=0.2, epsilon=0.1)

        cpp_sim = SEIZSimulator(use_cpp=True).solve_stochastic(
            state, params, t_span=(0.0, 60.0), num_points=241, num_simulations=1000, seed=123
        )
        assert "1000 C++ parallel simulations" in cpp_sim.message
        assert cpp_sim.p5.shape == (4, 241)
        assert cpp_sim.p50.shape == (4, 241)
        assert cpp_sim.p95.shape == (4, 241)

        for comp in ("S", "E", "I", "Z"):
            c_p5 = cpp_sim[f"{comp}_p5"]
            c_p50 = cpp_sim[f"{comp}_p50"]
            c_p95 = cpp_sim[f"{comp}_p95"]
            assert np.all(0.0 <= c_p5)
            assert np.all(c_p5 <= c_p50 + 1e-12)
            assert np.all(c_p50 <= c_p95 + 1e-12)
            assert np.all(c_p95 <= float(state.N))

        # Early/mid-stage trajectory variance should be strictly positive.
        assert np.any(cpp_sim.I_p95 > cpp_sim.I_p5)
        assert np.any(cpp_sim.Z_p95 > cpp_sim.Z_p5)

        # Verify pure-Python stochastic fallback produces consistent percentile bands.
        py_sim = SEIZSimulator(use_cpp=False, stochastic=True, num_simulations=50, seed=123).solve(
            state, params, t_span=(0.0, 30.0), num_points=61
        )
        assert "Python fallback" in py_sim.message
        assert np.all(py_sim.I_p5 <= py_sim.I_p50 + 1e-12)
        assert np.all(py_sim.I_p50 <= py_sim.I_p95 + 1e-12)
        assert np.all(py_sim.Z_p5 <= py_sim.Z_p50 + 1e-12)
        assert np.all(py_sim.Z_p50 <= py_sim.Z_p95 + 1e-12)


class TestEigenVectorizedBatchSolver:
    """Tests and <100ms benchmark for Eigen3-vectorized batched RK4 integration."""

    def test_eigen_batch_derivatives_and_trajectories_match_single_solver(self) -> None:
        state = InitialState(S0=950, E0=25, I0=15, Z0=10)
        y0 = state.to_array()

        params_list = [
            [0.6, 0.25, 0.4, 0.3, 0.2, 0.1],
            [0.45, 0.5, 0.2, 0.15, 0.1, 0.05],
            [0.8, 0.1, 0.7, 0.4, 0.3, 0.12],
            [0.2, 0.9, 0.1, 0.05, 0.05, 0.02],
            [0.55, 0.35, 0.45, 0.25, 0.18, 0.08],
            [0.7, 0.4, 0.6, 0.35, 0.22, 0.09],
        ]
        params_batch = np.asarray(params_list, dtype=np.float64)
        y_batch = np.repeat(y0[np.newaxis, :], len(params_list), axis=0)

        # Verify vectorized Eigen derivatives match scalar Python derivatives.
        batch_deriv = seiz_cpp.seiz_derivatives_batch(y_batch, params_batch)
        assert batch_deriv.shape == (6, 4)
        for idx, p_vals in enumerate(params_list):
            p_model = SEIZParameters(
                beta=p_vals[0],
                p=p_vals[1],
                l=p_vals[2],
                b=p_vals[3],
                rho=p_vals[4],
                epsilon=p_vals[5],
            )
            expected_dy = SEIZSimulator._seiz_derivatives(0.0, y0, p_model)
            np.testing.assert_allclose(batch_deriv[idx], expected_dy, rtol=1e-12, atol=1e-12)

        # Verify 3D output [N, time_steps, 4] via solve_rk4 overload and SEIZSimulator.solve_batch.
        num_points = 101
        t_span = (0.0, 20.0)
        dt = 20.0 / (num_points - 1)

        traj_overload = seiz_cpp.solve_rk4(y0, params_batch, dt, num_points - 1, 0.0)
        traj_batch = SEIZSimulator(use_cpp=True).solve_batch(
            state, params_batch, t_span=t_span, num_points=num_points
        )
        traj_via_solve = SEIZSimulator(use_cpp=True).solve(
            state, params_batch, t_span=t_span, num_points=num_points
        )
        traj_numpy_fallback = SEIZSimulator(use_cpp=False).solve_batch(
            state, params_batch, t_span=t_span, num_points=num_points
        )

        assert traj_overload.shape == (6, num_points, 4)
        assert traj_batch.shape == (6, num_points, 4)
        assert isinstance(traj_via_solve, np.ndarray)
        np.testing.assert_allclose(traj_overload, traj_batch, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(traj_batch, traj_via_solve, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(traj_batch, traj_numpy_fallback, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(traj_batch.sum(axis=-1), float(state.N), rtol=0.0, atol=1e-6)

        single_sim = SEIZSimulator(use_cpp=True)
        for idx, p_vals in enumerate(params_list):
            p_model = SEIZParameters(
                beta=p_vals[0],
                p=p_vals[1],
                l=p_vals[2],
                b=p_vals[3],
                rho=p_vals[4],
                epsilon=p_vals[5],
            )
            single_res = single_sim.solve(state, p_model, t_span=t_span, num_points=num_points)
            np.testing.assert_allclose(traj_batch[idx].T, single_res.y, rtol=1e-10, atol=1e-10)

    def test_benchmark_10000_parameter_combinations_under_100ms(self) -> None:
        state = InitialState(S0=9900, E0=50, I0=30, Z0=20)
        rng = np.random.default_rng(2026)

        n_combos = 10_000
        params_batch = np.column_stack(
            [
                rng.uniform(0.1, 0.9, size=n_combos),   # beta
                rng.uniform(0.0, 1.0, size=n_combos),   # p
                rng.uniform(0.0, 1.0, size=n_combos),   # l
                rng.uniform(0.05, 0.6, size=n_combos),  # b
                rng.uniform(0.05, 0.5, size=n_combos),  # rho
                rng.uniform(0.01, 0.3, size=n_combos),  # epsilon
            ]
        )

        y0 = state.to_array()
        num_points = 100
        dt = 20.0 / (num_points - 1)
        max_steps = num_points - 1

        # Warmup call to ensure CPU caches and worker pools are initialized.
        _ = seiz_cpp.solve_rk4_batch(y0, params_batch[:64], dt, max_steps, 0.0)

        t_start = time.perf_counter()
        trajectories = seiz_cpp.solve_rk4_batch(y0, params_batch, dt, max_steps, 0.0)
        elapsed_s = time.perf_counter() - t_start
        elapsed_ms = elapsed_s * 1000.0

        assert trajectories.shape == (10_000, 100, 4)
        np.testing.assert_allclose(trajectories.sum(axis=-1), float(state.N), rtol=0.0, atol=1e-6)

        structlog.get_logger(__name__).info(
            "benchmark_10000_parameter_combinations",
            batch_size=n_combos,
            time_steps=num_points,
            elapsed_ms=round(elapsed_ms, 2),
        )
        assert elapsed_ms < 100.0


class TestObservabilityAndMetrics:
    """Tests for structured JSON logging and Prometheus instrumentation in the math core."""

    @staticmethod
    def _collect_metric_sample(
        metric: Counter | Histogram | Summary,
        suffix: str,
    ) -> float:
        total = 0.0
        for family in metric.collect():
            for sample in family.samples:
                if sample.name.endswith(suffix):
                    total += float(sample.value)
        return total

    def test_cpp_simulation_emits_structured_json_and_records_prometheus_metrics(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        configure_structlog()
        state = InitialState(S0=950, E0=25, I0=15, Z0=10)
        params = SEIZParameters(beta=0.6, p=0.3, l=0.4, b=0.25, rho=0.2, epsilon=0.1)

        hist_count_before = self._collect_metric_sample(CPP_INTEGRATION_DURATION_MS, "_count")
        hist_sum_before = self._collect_metric_sample(CPP_INTEGRATION_DURATION_MS, "_sum")
        summ_count_before = self._collect_metric_sample(CPP_INTEGRATION_SUMMARY_MS, "_count")
        summ_sum_before = self._collect_metric_sample(CPP_INTEGRATION_SUMMARY_MS, "_sum")

        capsys.readouterr()
        res = SEIZSimulator(use_cpp=True).solve(
            state,
            params,
            t_span=(0.0, 20.0),
            num_points=101,
            task_id="test-sim-task-123",
        )
        assert res.success is True

        captured = capsys.readouterr().out.strip().splitlines()
        assert len(captured) >= 1
        record = json.loads(captured[-1])
        assert record["event"] == "simulation_completed"
        assert record["task_id"] == "test-sim-task-123"
        assert isinstance(record["execution_time_ms"], float)
        assert record["execution_time_ms"] >= 0.0
        assert record["solver_status"] == "completed"
        assert record["level"] == "info"
        assert "timestamp" in record

        hist_count_after = self._collect_metric_sample(CPP_INTEGRATION_DURATION_MS, "_count")
        hist_sum_after = self._collect_metric_sample(CPP_INTEGRATION_DURATION_MS, "_sum")
        summ_count_after = self._collect_metric_sample(CPP_INTEGRATION_SUMMARY_MS, "_count")
        summ_sum_after = self._collect_metric_sample(CPP_INTEGRATION_SUMMARY_MS, "_sum")

        assert hist_count_after == hist_count_before + 1.0
        assert hist_sum_after > hist_sum_before
        assert summ_count_after == summ_count_before + 1.0
        assert summ_sum_after > summ_sum_before

    def test_optimization_failure_increments_prometheus_counter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        estimator = ParameterEstimator()
        t_obs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        i_obs = np.array([15.0, 22.0, 31.0, 44.0], dtype=np.float64)
        z_obs = np.array([10.0, 12.0, 15.0, 19.0], dtype=np.float64)
        state = InitialState(S0=950, E0=25, I0=15, Z0=10)

        before = self._collect_metric_sample(OPTIMIZATION_FAILURES_TOTAL, "_total")

        def fail_linalg(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise np.linalg.LinAlgError("Simulated singular Hessian")

        monkeypatch.setattr("src.simulation.calibration.minimize", fail_linalg)
        with pytest.raises(OptimizationConvergenceError):
            estimator.fit_to_data(t_obs, i_obs, z_obs, state)

        after = self._collect_metric_sample(OPTIMIZATION_FAILURES_TOTAL, "_total")
        assert after == before + 1.0

    def test_no_standard_logging_or_print_in_source(self) -> None:
        root = Path(__file__).resolve().parents[1]
        py_files = [root / "main.py", *(root / "src").rglob("*.py")]
        for file_path in py_files:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = {alias.name for alias in node.names}
                    assert "logging" not in imported, f"Found stdlib logging in {file_path}"
                elif isinstance(node, ast.ImportFrom):
                    assert node.module != "logging", f"Found stdlib logging in {file_path}"
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id != "print", f"Found print() call in {file_path}"
