"""Inverse-problem parameter calibration for the SEIZ disinformation ODE model."""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Sequence
from typing import NoReturn

import numpy as np
import structlog
from prometheus_client import REGISTRY, Counter, Histogram
from pydantic import ValidationError
from scipy.linalg import LinAlgError as SciPyLinAlgError
from scipy.optimize import OptimizeWarning, minimize

from src.models.schemas import InitialState, SEIZParameters
from src.simulation.engine import SEIZSimulator, configure_structlog, get_or_create_metric

configure_structlog()
logger = structlog.get_logger(__name__)

_CALIBRATION_DURATION_BUCKETS: tuple[float, ...] = (
    1.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    5000.0,
)

OPTIMIZATION_FAILURES_TOTAL: Counter = get_or_create_metric(
    "seiz_optimization_failures_total",
    Counter,
    lambda: Counter(
        "seiz_optimization_failures_total",
        "Total number of SEIZ parameter calibration optimization failures",
        registry=REGISTRY,
    ),
)
CALIBRATION_DURATION_MS: Histogram = get_or_create_metric(
    "seiz_calibration_duration_ms",
    Histogram,
    lambda: Histogram(
        "seiz_calibration_duration_ms",
        "Millisecond duration of SEIZ parameter calibration runs",
        buckets=_CALIBRATION_DURATION_BUCKETS,
        registry=REGISTRY,
    ),
)


def _raise_convergence_error(
    message: str,
    exc: Exception | None = None,
) -> NoReturn:
    """Increment optimization failure counter, log structured error, and raise."""
    OPTIMIZATION_FAILURES_TOTAL.inc()
    logger.error("optimization_convergence_failed", error=message)
    if exc is not None:
        raise OptimizationConvergenceError(message) from exc
    raise OptimizationConvergenceError(message)

PARAM_NAMES: tuple[str, ...] = ("beta", "p", "l", "b", "rho", "epsilon")
DEFAULT_BOUNDS: tuple[tuple[float, float | None], ...] = (
    (0.0, None),   # beta >= 0
    (0.0, 1.0),    # p in [0, 1]
    (0.0, 1.0),    # l in [0, 1]
    (0.0, None),   # b >= 0
    (0.0, None),   # rho >= 0
    (0.0, None),   # epsilon >= 0
)
DEFAULT_INITIAL_GUESS: tuple[float, ...] = (0.5, 0.5, 0.5, 0.2, 0.2, 0.1)
_BOUNDARY_EPS: float = 1e-12
_FAILURE_PENALTY: float = 1e18


class OptimizationConvergenceError(RuntimeError):
    """Raised when calibration fails due to matrix inversion or fatal numerical failure."""


class ParameterEstimator:
    """Calibrates SEIZ model parameters to observed $(I(t), Z(t))$ time-series data."""

    def __init__(self, simulator: SEIZSimulator | None = None) -> None:
        """Initialize the estimator with a SEIZSimulator instance."""
        self.simulator: SEIZSimulator = simulator if simulator is not None else SEIZSimulator()

    @staticmethod
    def _validate_initial_state(initial_state: InitialState | dict[str, int]) -> InitialState:
        """Validate and coerce initial compartment state."""
        if isinstance(initial_state, dict):
            return InitialState.model_validate(initial_state)
        if isinstance(initial_state, InitialState):
            return initial_state
        raise TypeError("initial_state must be an InitialState instance or a valid dict.")

    @staticmethod
    def _validate_observations(
        t_observed: np.ndarray | Sequence[float],
        I_observed: np.ndarray | Sequence[float],
        Z_observed: np.ndarray | Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate and coerce observed time and compartment arrays."""
        t_obs = np.asarray(t_observed, dtype=np.float64)
        i_obs = np.asarray(I_observed, dtype=np.float64)
        z_obs = np.asarray(Z_observed, dtype=np.float64)

        if t_obs.ndim != 1 or i_obs.ndim != 1 or z_obs.ndim != 1:
            raise ValueError("t_observed, I_observed, and Z_observed must be 1D sequences.")
        if t_obs.size < 2:
            raise ValueError("Observed time series must contain at least 2 time points.")
        if not (t_obs.shape == i_obs.shape == z_obs.shape):
            raise ValueError("t_observed, I_observed, and Z_observed must have identical lengths.")
        if not (
            np.all(np.isfinite(t_obs))
            and np.all(np.isfinite(i_obs))
            and np.all(np.isfinite(z_obs))
        ):
            raise ValueError("Observed arrays must contain only finite numeric values.")
        if np.any(np.diff(t_obs) <= 0.0):
            raise ValueError("t_observed must be strictly increasing.")

        return t_obs, i_obs, z_obs

    @staticmethod
    def _array_to_params(params_array: np.ndarray | Sequence[float]) -> SEIZParameters:
        """Convert a 6-element parameter vector into a validated SEIZParameters model."""
        arr = np.asarray(params_array, dtype=np.float64)
        if arr.shape != (len(PARAM_NAMES),):
            raise ValueError(
                f"params_array must have shape ({len(PARAM_NAMES)},), got {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("All elements in params_array must be finite.")

        values: dict[str, float] = {}
        for idx, name in enumerate(PARAM_NAMES):
            val = float(arr[idx])
            # Clamp negligible floating-point overshoot from bounded optimization at 0 or 1.
            if -_BOUNDARY_EPS <= val < 0.0:
                val = 0.0
            if name in ("p", "l") and 1.0 < val <= 1.0 + _BOUNDARY_EPS:
                val = 1.0
            values[name] = val

        return SEIZParameters(**values)

    @staticmethod
    def _prepare_bounds(
        bounds: Sequence[tuple[float, float | None]] | None,
    ) -> tuple[tuple[tuple[float, float | None], ...], np.ndarray, np.ndarray]:
        """Validate parameter bounds and construct lower/upper clipping arrays."""
        active_bounds = DEFAULT_BOUNDS if bounds is None else tuple(bounds)
        if len(active_bounds) != len(PARAM_NAMES):
            raise ValueError(
                f"bounds must contain exactly {len(PARAM_NAMES)} (low, high) pairs."
            )

        low_vals: list[float] = []
        high_vals: list[float] = []
        for low, high in active_bounds:
            low_f = -np.inf if low is None else float(low)
            high_f = np.inf if high is None else float(high)
            if math.isnan(low_f) or math.isnan(high_f) or low_f > high_f:
                raise ValueError(f"Invalid parameter bound pair: ({low}, {high}).")
            low_vals.append(low_f)
            high_vals.append(high_f)

        return (
            active_bounds,
            np.asarray(low_vals, dtype=np.float64),
            np.asarray(high_vals, dtype=np.float64),
        )

    @staticmethod
    def _prepare_initial_guess(
        initial_guess: np.ndarray | Sequence[float] | SEIZParameters | None,
        low_bounds: np.ndarray,
        high_bounds: np.ndarray,
    ) -> np.ndarray:
        """Convert initial_guess to a 6-element float64 vector projected into bounds."""
        if initial_guess is None:
            x0 = np.array(DEFAULT_INITIAL_GUESS, dtype=np.float64)
        elif isinstance(initial_guess, SEIZParameters):
            x0 = np.array(
                [getattr(initial_guess, name) for name in PARAM_NAMES],
                dtype=np.float64,
            )
        else:
            x0 = np.asarray(initial_guess, dtype=np.float64)
            if x0.shape != (len(PARAM_NAMES),):
                raise ValueError(
                    f"initial_guess must have shape ({len(PARAM_NAMES)},), got {x0.shape}."
                )
            if not np.all(np.isfinite(x0)):
                raise ValueError("All elements in initial_guess must be finite.")

        return np.clip(x0, low_bounds, high_bounds)

    def _objective_function(
        self,
        params_array: np.ndarray | Sequence[float],
        t_observed: np.ndarray | Sequence[float],
        I_observed: np.ndarray | Sequence[float],
        Z_observed: np.ndarray | Sequence[float],
        initial_state: InitialState | dict[str, int],
    ) -> float:
        """Compute SSE between simulated and observed I(t), Z(t) over t_observed."""
        t_obs, i_obs, z_obs = self._validate_observations(t_observed, I_observed, Z_observed)
        validated_state = self._validate_initial_state(initial_state)

        arr = np.asarray(params_array, dtype=np.float64)
        if arr.shape != (len(PARAM_NAMES),):
            raise ValueError(
                f"params_array must have shape ({len(PARAM_NAMES)},), got {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            return _FAILURE_PENALTY

        try:
            params = self._array_to_params(arr)
            sim_result = self.simulator.solve(
                state=validated_state,
                params=params,
                t_span=(float(t_obs[0]), float(t_obs[-1])),
                t_eval=t_obs,
                _log_simulation=False,
            )
        except (ValidationError, RuntimeError, ValueError):
            return _FAILURE_PENALTY

        err_i = sim_result.I - i_obs
        err_z = sim_result.Z - z_obs
        sse = float(np.sum(err_i * err_i) + np.sum(err_z * err_z))
        if not math.isfinite(sse):
            return _FAILURE_PENALTY
        return sse

    @staticmethod
    def _validate_optimization_result(res: object) -> np.ndarray:
        """Verify solution finiteness and inverse-Hessian matrix invertibility."""
        x_opt = getattr(res, "x", None)
        fun_val = getattr(res, "fun", None)
        if x_opt is None or fun_val is None:
            _raise_convergence_error(
                "Optimization failed: solver did not return a solution vector or objective value."
            )

        x_arr = np.asarray(x_opt, dtype=np.float64)
        if x_arr.shape != (len(PARAM_NAMES),) or not np.all(np.isfinite(x_arr)):
            _raise_convergence_error(
                "Optimization failed: solution vector contains non-finite values."
            )

        fun_f = float(fun_val)
        if not math.isfinite(fun_f) or fun_f >= _FAILURE_PENALTY:
            _raise_convergence_error(
                "Optimization failed: objective value is non-finite or all trial "
                "integrations failed."
            )

        hess_inv = getattr(res, "hess_inv", None)
        if hess_inv is not None:
            try:
                hess_dense = hess_inv.todense() if hasattr(hess_inv, "todense") else hess_inv
                hess_mat = np.asarray(hess_dense, dtype=np.float64)
            except (np.linalg.LinAlgError, SciPyLinAlgError, ValueError, TypeError) as exc:
                _raise_convergence_error(
                    "Optimization failed: could not materialize approximate inverse "
                    "Hessian matrix.",
                    exc=exc,
                )

            expected_shape = (len(PARAM_NAMES), len(PARAM_NAMES))
            if hess_mat.shape != expected_shape or not np.all(np.isfinite(hess_mat)):
                _raise_convergence_error(
                    "Optimization failed: approximate inverse Hessian matrix is invalid or "
                    "non-finite."
                )

            try:
                inv_hess = np.linalg.inv(hess_mat)
            except (np.linalg.LinAlgError, SciPyLinAlgError) as exc:
                _raise_convergence_error(
                    "Optimization failed: matrix inversion of the approximate inverse "
                    "Hessian failed.",
                    exc=exc,
                )

            if not np.all(np.isfinite(inv_hess)):
                _raise_convergence_error(
                    "Optimization failed: inverted Hessian matrix contains non-finite values."
                )

        return x_arr

    def fit_to_data(
        self,
        t_observed: np.ndarray | Sequence[float],
        I_observed: np.ndarray | Sequence[float],
        Z_observed: np.ndarray | Sequence[float],
        initial_state: InitialState | dict[str, int],
        initial_guess: np.ndarray | Sequence[float] | SEIZParameters | None = None,
        bounds: Sequence[tuple[float, float | None]] | None = None,
        maxiter: int = 500,
        ftol: float = 1e-9,
        gtol: float = 1e-6,
    ) -> SEIZParameters:
        """Fit SEIZ parameters to observed (I, Z) time series via bounded L-BFGS-B optimization."""
        if maxiter < 1:
            raise ValueError("maxiter must be >= 1.")
        if not math.isfinite(ftol) or ftol <= 0.0:
            raise ValueError("ftol must be a finite positive float.")
        if not math.isfinite(gtol) or gtol <= 0.0:
            raise ValueError("gtol must be a finite positive float.")

        t_obs, i_obs, z_obs = self._validate_observations(t_observed, I_observed, Z_observed)
        validated_state = self._validate_initial_state(initial_state)
        active_bounds, low_bounds, high_bounds = self._prepare_bounds(bounds)
        x0 = self._prepare_initial_guess(initial_guess, low_bounds, high_bounds)

        t_cal_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", OptimizeWarning)
            warnings.simplefilter("always", RuntimeWarning)
            warnings.simplefilter("always", UserWarning)
            try:
                res = minimize(
                    fun=self._objective_function,
                    x0=x0,
                    args=(t_obs, i_obs, z_obs, validated_state),
                    method="L-BFGS-B",
                    bounds=active_bounds,
                    options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
                )
            except (np.linalg.LinAlgError, SciPyLinAlgError) as exc:
                _raise_convergence_error(
                    f"Optimization failed due to linear algebra / matrix inversion error: {exc}",
                    exc=exc,
                )

        for w in caught_warnings:
            logger.warning(
                "optimization_warning",
                category=w.category.__name__,
                warning_message=str(w.message),
            )

        x_arr = self._validate_optimization_result(res)
        try:
            params = self._array_to_params(x_arr)
        except ValidationError as exc:
            _raise_convergence_error(
                f"Optimization produced invalid SEIZ parameters: {exc}",
                exc=exc,
            )

        elapsed_ms = (time.perf_counter() - t_cal_start) * 1000.0
        CALIBRATION_DURATION_MS.observe(elapsed_ms)
        logger.info(
            "calibration_completed",
            execution_time_ms=round(elapsed_ms, 4),
            solver_status="completed",
        )
        return params
