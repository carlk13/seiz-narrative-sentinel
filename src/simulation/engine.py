"""SEIZ epidemiological ODE simulation engine."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np
import structlog
from prometheus_client import REGISTRY, Counter, Histogram, Summary
from scipy.integrate import solve_ivp

from src.models.schemas import InitialState, SEIZParameters

_CPP_DURATION_BUCKETS: tuple[float, ...] = (
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
)


def configure_structlog() -> None:
    """Configure structlog for machine-readable JSON logging with contextvars."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_or_create_metric[MetricT: (Counter, Histogram, Summary)](
    name: str,
    metric_cls: type[MetricT],
    factory: Callable[[], MetricT],
) -> MetricT:
    """Return an existing Prometheus metric from REGISTRY or construct and register it."""
    existing = REGISTRY._names_to_collectors.get(name)
    if isinstance(existing, metric_cls):
        return existing
    return factory()


configure_structlog()
logger = structlog.get_logger(__name__)

CPP_INTEGRATION_DURATION_MS: Histogram = get_or_create_metric(
    "seiz_cpp_integration_duration_ms",
    Histogram,
    lambda: Histogram(
        "seiz_cpp_integration_duration_ms",
        "Exact millisecond duration of the C++ SEIZ numerical integration step",
        buckets=_CPP_DURATION_BUCKETS,
        registry=REGISTRY,
    ),
)
CPP_INTEGRATION_SUMMARY_MS: Summary = get_or_create_metric(
    "seiz_cpp_integration_summary_ms",
    Summary,
    lambda: Summary(
        "seiz_cpp_integration_summary_ms",
        "Summary of millisecond duration of the C++ SEIZ numerical integration step",
        registry=REGISTRY,
    ),
)


def _record_cpp_duration_ms(duration_ms: float) -> None:
    """Record a C++ integration step duration in both Histogram and Summary metrics."""
    CPP_INTEGRATION_DURATION_MS.observe(duration_ms)
    CPP_INTEGRATION_SUMMARY_MS.observe(duration_ms)

_cpp_engine: Any = None
try:
    import src.cpp_core.seiz_cpp as _cpp_module

    _cpp_engine = _cpp_module
    HAS_CPP_CORE = True
except (ImportError, OSError):
    _cpp_engine = None
    HAS_CPP_CORE = False

if TYPE_CHECKING:

    class _OdeResultBase(dict[str, Any]):
        t: np.ndarray
        y: np.ndarray
        S: np.ndarray
        E: np.ndarray
        I: np.ndarray  # noqa: E741
        Z: np.ndarray
        N: np.ndarray
        p5: dict[str, np.ndarray]
        p50: dict[str, np.ndarray]
        p95: dict[str, np.ndarray]
        success: bool
        status: int
        message: str
        nfev: int
        njev: int
        nlu: int

        def __getattr__(self, name: str) -> Any: ...
        def __setattr__(self, name: str, value: Any) -> None: ...
else:
    from scipy.integrate._ivp.ivp import OdeResult as _OdeResultBase


class SimulationResult(_OdeResultBase):
    """ODE integration result exposing both dictionary and attribute access."""


class SEIZSimulator:
    """Numerical ODE simulator for the SEIZ disinformation epidemiological model."""

    def __init__(
        self,
        use_cpp: bool | None = None,
        stochastic: bool = False,
        num_simulations: int = 1000,
        seed: int = 42,
    ) -> None:
        """Initialize the simulator, preferring the compiled C++ core when available."""
        if num_simulations < 1:
            raise ValueError("num_simulations must be >= 1.")
        if seed < 0:
            raise ValueError("seed must be >= 0.")
        self.use_cpp: bool | None = use_cpp
        self.stochastic: bool = stochastic
        self.num_simulations: int = num_simulations
        self.seed: int = seed

    @staticmethod
    def _seiz_derivatives(
        t: float,
        y: np.ndarray | Sequence[float],
        params: SEIZParameters,
    ) -> np.ndarray:
        """Compute normalized SEIZ compartment derivatives [dS/dt, dE/dt, dI/dt, dZ/dt]."""
        del t  # Autonomous ODE system; time does not enter explicitly.
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.shape != (4,):
            raise ValueError(f"State vector y must have shape (4,), got {y_arr.shape}.")

        S, E, I, Z = (  # noqa: E741
            float(y_arr[0]),
            float(y_arr[1]),
            float(y_arr[2]),
            float(y_arr[3]),
        )
        N = S + E + I + Z
        if not math.isfinite(N) or N <= 0.0:
            raise ValueError("Total population N must be finite and strictly positive.")

        si_flow = params.beta * S * I / N
        sz_flow = params.b * S * Z / N
        ei_flow = params.rho * E * I / N
        incubation_flow = params.epsilon * E

        dS_dt = -si_flow - sz_flow
        dE_dt = (1.0 - params.p) * si_flow + (1.0 - params.l) * sz_flow - ei_flow - incubation_flow
        dI_dt = params.p * si_flow + ei_flow + incubation_flow
        dZ_dt = params.l * sz_flow

        return np.array([dS_dt, dE_dt, dI_dt, dZ_dt], dtype=np.float64)

    @staticmethod
    def _validate_inputs(
        state: InitialState | dict[str, int],
        params: SEIZParameters | dict[str, float],
    ) -> tuple[InitialState, SEIZParameters]:
        """Validate and coerce simulation state and parameter models."""
        if isinstance(state, dict):
            validated_state = InitialState.model_validate(state)
        elif isinstance(state, InitialState):
            validated_state = state
        else:
            raise TypeError("state must be an InitialState instance or a valid dict.")

        if isinstance(params, dict):
            validated_params = SEIZParameters.model_validate(params)
        elif isinstance(params, SEIZParameters):
            validated_params = params
        else:
            raise TypeError("params must be a SEIZParameters instance or a valid dict.")

        return validated_state, validated_params

    @staticmethod
    def _resolve_time_grid(
        t_span: tuple[float, float] | Sequence[float] | np.ndarray,
        t_eval: np.ndarray | Sequence[float] | None,
        num_points: int,
    ) -> tuple[tuple[float, float], np.ndarray]:
        """Normalize t_span and t_eval into a validated (t0, tf) tuple and 1D time grid."""
        if num_points < 2:
            raise ValueError("num_points must be at least 2.")

        span_arr = np.asarray(t_span, dtype=np.float64)
        if span_arr.ndim != 1 or span_arr.size < 2:
            raise ValueError("t_span must be a 1D sequence of at least 2 time points.")
        if not np.all(np.isfinite(span_arr)):
            raise ValueError("All time points in t_span must be finite.")

        if span_arr.size == 2:
            t0, tf = float(span_arr[0]), float(span_arr[1])
            if tf < t0:
                raise ValueError(f"Invalid t_span: end time ({tf}) must be >= start time ({t0}).")

            if t_eval is None:
                if tf == t0:
                    t_eval_arr = np.array([t0], dtype=np.float64)
                else:
                    t_eval_arr = np.linspace(t0, tf, num_points, dtype=np.float64)
            else:
                t_eval_arr = np.asarray(t_eval, dtype=np.float64)
                if t_eval_arr.ndim != 1 or t_eval_arr.size < 1:
                    raise ValueError("t_eval must be a non-empty 1D sequence.")
                if not np.all(np.isfinite(t_eval_arr)):
                    raise ValueError("All time points in t_eval must be finite.")
                if np.any(t_eval_arr < t0) or np.any(t_eval_arr > tf):
                    raise ValueError("All values in t_eval must lie within [t0, tf].")
                if t_eval_arr.size > 1 and np.any(np.diff(t_eval_arr) <= 0.0):
                    raise ValueError("Values in t_eval must be strictly increasing.")

            return (t0, tf), t_eval_arr

        if t_eval is not None:
            raise ValueError("t_span must be a (t0, tf) pair when t_eval is also provided.")
        if np.any(np.diff(span_arr) <= 0.0):
            raise ValueError("Explicit time grid passed via t_span must be strictly increasing.")

        return (float(span_arr[0]), float(span_arr[-1])), span_arr

    @staticmethod
    def _build_result(
        t: np.ndarray,
        y: np.ndarray,
        success: bool,
        status: int,
        message: str,
        nfev: int = 0,
        njev: int = 0,
        nlu: int = 0,
        p5: np.ndarray | None = None,
        p50: np.ndarray | None = None,
        p95: np.ndarray | None = None,
        n_arr: np.ndarray | None = None,
    ) -> SimulationResult:
        """Construct a SimulationResult dictionary/attribute view from state trajectories."""
        p5_arr = y if p5 is None else p5
        p50_arr = y if p50 is None else p50
        p95_arr = y if p95 is None else p95
        total_n = y.sum(axis=0) if n_arr is None else n_arr

        return SimulationResult(
            t=t,
            y=y,
            S=y[0],
            E=y[1],
            I=y[2],
            Z=y[3],
            N=total_n,
            p5=p5_arr,
            p50=p50_arr,
            p95=p95_arr,
            S_p5=p5_arr[0],
            S_p50=p50_arr[0],
            S_p95=p95_arr[0],
            E_p5=p5_arr[1],
            E_p50=p50_arr[1],
            E_p95=p95_arr[1],
            I_p5=p5_arr[2],
            I_p50=p50_arr[2],
            I_p95=p95_arr[2],
            Z_p5=p5_arr[3],
            Z_p50=p50_arr[3],
            Z_p95=p95_arr[3],
            success=success,
            status=status,
            message=message,
            nfev=nfev,
            njev=njev,
            nlu=nlu,
        )

    @staticmethod
    def _interp_band_to_grid(
        t_src: np.ndarray,
        arr_4xn: np.ndarray,
        t_target: np.ndarray,
    ) -> np.ndarray:
        """Linearly interpolate a (4, M) trajectory array onto target timestamps."""
        out = np.empty((4, t_target.size), dtype=np.float64)
        for comp in range(4):
            out[comp] = np.interp(t_target, t_src, arr_4xn[comp])
        return out

    @staticmethod
    def _run_cpp_rk4(
        curr_state: tuple[float, float, float, float],
        param_tuple: tuple[float, float, float, float, float, float],
        dt: float,
        steps: int,
        t0: float,
    ) -> dict[str, Any]:
        """Invoke compiled C++ RK4 solver and record integration step duration in ms."""
        assert _cpp_engine is not None
        t_cpp_start = time.perf_counter()
        raw_res = _cpp_engine.solve_rk4(curr_state, param_tuple, dt, steps, t0)
        _record_cpp_duration_ms((time.perf_counter() - t_cpp_start) * 1000.0)
        return cast(dict[str, Any], raw_res)

    @staticmethod
    def _run_cpp_gillespie(
        curr_state: tuple[float, float, float, float],
        param_tuple: tuple[float, float, float, float, float, float],
        dt: float,
        steps: int,
        t0: float,
        num_simulations: int,
        seed: int,
    ) -> dict[str, Any]:
        """Invoke compiled C++ Gillespie solver and record integration step duration in ms."""
        assert _cpp_engine is not None
        t_cpp_start = time.perf_counter()
        raw_res = _cpp_engine.solve_gillespie(
            curr_state, param_tuple, dt, steps, t0, num_simulations, seed
        )
        _record_cpp_duration_ms((time.perf_counter() - t_cpp_start) * 1000.0)
        return cast(dict[str, Any], raw_res)

    def _solve_gillespie_cpp(
        self,
        state: InitialState,
        params: SEIZParameters,
        span: tuple[float, float],
        t_eval_arr: np.ndarray,
        num_simulations: int,
        seed: int,
    ) -> SimulationResult:
        """Execute parallel Gillespie stochastic simulations via the compiled C++ module."""
        assert _cpp_engine is not None
        curr_state = (float(state.S0), float(state.E0), float(state.I0), float(state.Z0))
        param_tuple = (
            params.beta,
            params.p,
            params.l,
            params.b,
            params.rho,
            params.epsilon,
        )
        t0 = span[0]
        n_points = int(t_eval_arr.size)

        if n_points == 1:
            dt_single = float(t_eval_arr[0] - t0)
            if dt_single > 0.0:
                cpp_res = self._run_cpp_gillespie(
                    curr_state, param_tuple, dt_single, 1, t0, num_simulations, seed
                )
                p5 = np.asarray(cpp_res["p5"], dtype=np.float64).reshape(4, 2)[:, -1:]
                p50 = np.asarray(cpp_res["p50"], dtype=np.float64).reshape(4, 2)[:, -1:]
                p95 = np.asarray(cpp_res["p95"], dtype=np.float64).reshape(4, 2)[:, -1:]
            else:
                y0_col = state.to_array()[:, np.newaxis]
                p5 = p50 = p95 = y0_col
            n_arr = np.full((1,), float(state.N), dtype=np.float64)
            return self._build_result(
                t=t_eval_arr,
                y=p50,
                success=True,
                status=0,
                message=(
                    f"Stochastic Gillespie integration completed "
                    f"({num_simulations} C++ parallel simulations)."
                ),
                p5=p5,
                p50=p50,
                p95=p95,
                n_arr=n_arr,
            )

        diffs = np.diff(t_eval_arr)
        dt = float((t_eval_arr[-1] - t_eval_arr[0]) / (n_points - 1))
        is_uniform_from_t0 = (
            abs(float(t_eval_arr[0]) - t0) <= 1e-12
            and bool(np.max(np.abs(diffs - dt)) <= 1e-12 + 1e-9 * abs(dt))
        )

        if is_uniform_from_t0:
            cpp_res = self._run_cpp_gillespie(
                curr_state, param_tuple, dt, n_points - 1, t0, num_simulations, seed
            )
            p5 = np.asarray(cpp_res["p5"], dtype=np.float64).reshape(4, n_points)
            p50 = np.asarray(cpp_res["p50"], dtype=np.float64).reshape(4, n_points)
            p95 = np.asarray(cpp_res["p95"], dtype=np.float64).reshape(4, n_points)
            n_arr = np.asarray(cpp_res["N"], dtype=np.float64)
        else:
            t_end = float(t_eval_arr[-1])
            grid_steps = max(500, n_points * 5)
            dt_grid = (t_end - t0) / grid_steps
            cpp_res = self._run_cpp_gillespie(
                curr_state, param_tuple, dt_grid, grid_steps, t0, num_simulations, seed
            )
            t_grid = np.asarray(cpp_res["t"], dtype=np.float64)
            p5_grid = np.asarray(cpp_res["p5"], dtype=np.float64).reshape(4, grid_steps + 1)
            p50_grid = np.asarray(cpp_res["p50"], dtype=np.float64).reshape(4, grid_steps + 1)
            p95_grid = np.asarray(cpp_res["p95"], dtype=np.float64).reshape(4, grid_steps + 1)
            p5 = self._interp_band_to_grid(t_grid, p5_grid, t_eval_arr)
            p50 = self._interp_band_to_grid(t_grid, p50_grid, t_eval_arr)
            p95 = self._interp_band_to_grid(t_grid, p95_grid, t_eval_arr)
            n_arr = np.full(n_points, float(state.N), dtype=np.float64)

        return self._build_result(
            t=t_eval_arr,
            y=p50,
            success=True,
            status=0,
            message=(
                f"Stochastic Gillespie integration completed "
                f"({num_simulations} C++ parallel simulations)."
            ),
            p5=p5,
            p50=p50,
            p95=p95,
            n_arr=n_arr,
        )

    def _solve_gillespie_python(
        self,
        state: InitialState,
        params: SEIZParameters,
        span: tuple[float, float],
        t_eval_arr: np.ndarray,
        num_simulations: int,
        seed: int,
    ) -> SimulationResult:
        """Pure-Python/NumPy Gillespie fallback when the compiled C++ core is unavailable."""
        rng = np.random.default_rng(seed)
        n_points = int(t_eval_arr.size)
        trajectories = np.empty((num_simulations, 4, n_points), dtype=np.float64)
        total_n = float(state.N)
        t0 = span[0]
        t_end = float(t_eval_arr[-1])

        beta, p_val, l_val = params.beta, params.p, params.l
        b_val, rho, epsilon = params.b, params.rho, params.epsilon

        for sim_idx in range(num_simulations):
            s, e, i, z = state.S0, state.E0, state.I0, state.Z0
            t_curr = t0
            step_idx = 0

            while step_idx < n_points and t_eval_arr[step_idx] <= t_curr:
                trajectories[sim_idx, :, step_idx] = (s, e, i, z)
                step_idx += 1

            while step_idx < n_points:
                si_rate = beta * s * i / total_n
                sz_rate = b_val * s * z / total_n
                ei_rate = rho * e * i / total_n
                inc_rate = epsilon * e

                a0 = p_val * si_rate
                a1 = (1.0 - p_val) * si_rate
                a2 = l_val * sz_rate
                a3 = (1.0 - l_val) * sz_rate
                a4 = ei_rate
                a5 = inc_rate
                a_tot = a0 + a1 + a2 + a3 + a4 + a5

                if a_tot <= 0.0:
                    while step_idx < n_points:
                        trajectories[sim_idx, :, step_idx] = (s, e, i, z)
                        step_idx += 1
                    break

                u1 = max(1e-15, 1.0 - float(rng.random()))
                tau = -math.log(u1) / a_tot
                t_next = t_curr + tau

                while step_idx < n_points and t_eval_arr[step_idx] <= t_next:
                    trajectories[sim_idx, :, step_idx] = (s, e, i, z)
                    step_idx += 1

                if t_next > t_end or step_idx >= n_points:
                    break

                t_curr = t_next
                r = float(rng.random()) * a_tot
                cum = a0
                if a0 > 0.0 and r < cum and s > 0:
                    s -= 1
                    i += 1
                elif a1 > 0.0 and r < (cum := cum + a1) and s > 0:
                    s -= 1
                    e += 1
                elif a2 > 0.0 and r < (cum := cum + a2) and s > 0:
                    s -= 1
                    z += 1
                elif a3 > 0.0 and r < (cum := cum + a3) and s > 0:
                    s -= 1
                    e += 1
                elif a4 > 0.0 and r < (cum + a4) and e > 0:
                    e -= 1
                    i += 1
                elif e > 0:
                    e -= 1
                    i += 1

        pcts = np.percentile(trajectories, [5.0, 50.0, 95.0], axis=0)
        p5, p50, p95 = pcts[0], pcts[1], pcts[2]
        n_arr = np.full(n_points, total_n, dtype=np.float64)
        return self._build_result(
            t=t_eval_arr,
            y=p50,
            success=True,
            status=0,
            message=(
                f"Stochastic Gillespie integration completed "
                f"({num_simulations} Python fallback simulations)."
            ),
            p5=p5,
            p50=p50,
            p95=p95,
            n_arr=n_arr,
        )

    def _solve_cpp(
        self,
        state: InitialState,
        params: SEIZParameters,
        span: tuple[float, float],
        t_eval_arr: np.ndarray,
    ) -> SimulationResult:
        """Integrate the SEIZ ODE system using the compiled C++ pybind11 RK4 module."""
        assert _cpp_engine is not None
        curr_state = (float(state.S0), float(state.E0), float(state.I0), float(state.Z0))
        param_tuple = (
            params.beta,
            params.p,
            params.l,
            params.b,
            params.rho,
            params.epsilon,
        )
        t0 = span[0]
        n_points = int(t_eval_arr.size)

        if n_points == 1:
            dt_single = float(t_eval_arr[0] - t0)
            if dt_single > 0.0:
                step_res = self._run_cpp_rk4(curr_state, param_tuple, dt_single, 1, t0)
                y_arr = np.array(
                    [
                        [step_res["S"][-1]],
                        [step_res["E"][-1]],
                        [step_res["I"][-1]],
                        [step_res["Z"][-1]],
                    ],
                    dtype=np.float64,
                )
            else:
                y_arr = state.to_array()[:, np.newaxis]
            return self._build_result(
                t=t_eval_arr,
                y=y_arr,
                success=True,
                status=0,
                message="Integration completed using C++ RK4 core.",
            )

        diffs = np.diff(t_eval_arr)
        dt = float((t_eval_arr[-1] - t_eval_arr[0]) / (n_points - 1))
        is_uniform = bool(np.max(np.abs(diffs - dt)) <= 1e-12 + 1e-9 * abs(dt))

        if is_uniform:
            t_start = float(t_eval_arr[0])
            if t_start > t0:
                pre_res = self._run_cpp_rk4(curr_state, param_tuple, t_start - t0, 1, t0)
                curr_state = (
                    float(pre_res["S"][-1]),
                    float(pre_res["E"][-1]),
                    float(pre_res["I"][-1]),
                    float(pre_res["Z"][-1]),
                )
            cpp_res = self._run_cpp_rk4(curr_state, param_tuple, dt, n_points - 1, t_start)
            y_arr = np.asarray(cpp_res["y_flat"], dtype=np.float64).reshape(4, n_points)
            return self._build_result(
                t=t_eval_arr,
                y=y_arr,
                success=True,
                status=0,
                message="Integration completed using C++ RK4 core.",
            )

        y_arr = np.empty((4, n_points), dtype=np.float64)
        t_curr = t0
        for idx, t_target in enumerate(t_eval_arr):
            t_target_f = float(t_target)
            if t_target_f > t_curr:
                step_res = self._run_cpp_rk4(
                    curr_state, param_tuple, t_target_f - t_curr, 1, t_curr
                )
                curr_state = (
                    float(step_res["S"][-1]),
                    float(step_res["E"][-1]),
                    float(step_res["I"][-1]),
                    float(step_res["Z"][-1]),
                )
                t_curr = t_target_f
            y_arr[:, idx] = curr_state

        return self._build_result(
            t=t_eval_arr,
            y=y_arr,
            success=True,
            status=0,
            message="Integration completed using C++ RK4 core.",
        )

    @overload
    def solve(
        self,
        state: InitialState | dict[str, int] | Sequence[float] | np.ndarray,
        params: SEIZParameters | dict[str, float],
        t_span: tuple[float, float] | Sequence[float] | np.ndarray,
        t_eval: np.ndarray | Sequence[float] | None = None,
        num_points: int = 500,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        stochastic: bool | None = None,
        num_simulations: int | None = None,
        seed: int | None = None,
        task_id: str | None = None,
        _log_simulation: bool = True,
    ) -> SimulationResult: ...

    @overload
    def solve(
        self,
        state: InitialState | dict[str, int] | Sequence[float] | np.ndarray,
        params: np.ndarray,
        t_span: tuple[float, float] | Sequence[float] | np.ndarray,
        t_eval: np.ndarray | Sequence[float] | None = None,
        num_points: int = 500,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        stochastic: bool | None = None,
        num_simulations: int | None = None,
        seed: int | None = None,
        task_id: str | None = None,
        _log_simulation: bool = True,
    ) -> np.ndarray: ...

    def solve(
        self,
        state: InitialState | dict[str, int] | Sequence[float] | np.ndarray,
        params: SEIZParameters | dict[str, float] | np.ndarray,
        t_span: tuple[float, float] | Sequence[float] | np.ndarray,
        t_eval: np.ndarray | Sequence[float] | None = None,
        num_points: int = 500,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        stochastic: bool | None = None,
        num_simulations: int | None = None,
        seed: int | None = None,
        task_id: str | None = None,
        _log_simulation: bool = True,
    ) -> SimulationResult | np.ndarray:
        """Integrate SEIZ ODEs (single run, Gillespie stochastic, or batched 2D parameters)."""
        resolved_task_id = str(
            task_id
            or structlog.contextvars.get_contextvars().get("task_id")
            or "direct-simulation"
        )
        t_start = time.perf_counter()

        try:
            if isinstance(params, np.ndarray) and params.ndim == 2:
                span_arr = np.asarray(t_span, dtype=np.float64)
                if span_arr.ndim != 1 or span_arr.size < 2:
                    raise ValueError("t_span must contain at least [t0, tf].")
                batch_out = self.solve_batch(
                    state=state,
                    params_batch=params,
                    t_span=(float(span_arr[0]), float(span_arr[-1])),
                    num_points=num_points,
                )
                if _log_simulation:
                    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 4)
                    logger.info(
                        "simulation_completed",
                        task_id=resolved_task_id,
                        execution_time_ms=elapsed_ms,
                        solver_status="completed",
                        status_code=0,
                    )
                return batch_out

            result = self._solve_single_or_stochastic(
                state=state,
                params=params,
                t_span=t_span,
                t_eval=t_eval,
                num_points=num_points,
                rtol=rtol,
                atol=atol,
                stochastic=stochastic,
                num_simulations=num_simulations,
                seed=seed,
            )
        except Exception as exc:
            if _log_simulation:
                elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 4)
                logger.error(
                    "simulation_failed",
                    task_id=resolved_task_id,
                    execution_time_ms=elapsed_ms,
                    solver_status="failed",
                    error=str(exc),
                )
            raise

        if _log_simulation:
            elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 4)
            logger.info(
                "simulation_completed",
                task_id=resolved_task_id,
                execution_time_ms=elapsed_ms,
                solver_status="completed" if result.success else "failed",
                status_code=int(result.status),
            )
        return result

    def _solve_single_or_stochastic(
        self,
        state: InitialState | dict[str, int] | Sequence[float] | np.ndarray,
        params: SEIZParameters | dict[str, float] | np.ndarray,
        t_span: tuple[float, float] | Sequence[float] | np.ndarray,
        t_eval: np.ndarray | Sequence[float] | None,
        num_points: int,
        rtol: float,
        atol: float,
        stochastic: bool | None,
        num_simulations: int | None,
        seed: int | None,
    ) -> SimulationResult:
        """Validate inputs and execute a single deterministic or stochastic simulation."""

        if not math.isfinite(rtol) or rtol <= 0.0:
            raise ValueError("rtol must be a finite positive float.")
        if not math.isfinite(atol) or atol <= 0.0:
            raise ValueError("atol must be a finite positive float.")

        is_stochastic = self.stochastic if stochastic is None else stochastic
        n_sims = self.num_simulations if num_simulations is None else num_simulations
        rng_seed = self.seed if seed is None else seed
        if n_sims < 1:
            raise ValueError("num_simulations must be >= 1.")
        if rng_seed < 0:
            raise ValueError("seed must be >= 0.")

        validated_state, validated_params = self._validate_inputs(state, params)  # type: ignore[arg-type]
        span, t_eval_arr = self._resolve_time_grid(t_span, t_eval, num_points)
        y0 = validated_state.to_array()

        if span[0] == span[1]:
            # SciPy's solve_ivp drops points when t0 == tf and t_eval is supplied;
            # on a zero-length interval the exact state equals y0 at t0.
            y_arr = np.repeat(y0[:, np.newaxis], t_eval_arr.size, axis=1)
            return self._build_result(
                t=t_eval_arr,
                y=y_arr,
                success=True,
                status=0,
                message="The solver successfully reached the end of the integration interval.",
            )

        cpp_active = (
            (self.use_cpp if self.use_cpp is not None else HAS_CPP_CORE)
            and (_cpp_engine is not None)
        )

        if is_stochastic:
            if cpp_active:
                return self._solve_gillespie_cpp(
                    validated_state, validated_params, span, t_eval_arr, n_sims, rng_seed
                )
            return self._solve_gillespie_python(
                validated_state, validated_params, span, t_eval_arr, n_sims, rng_seed
            )

        if cpp_active:
            return self._solve_cpp(validated_state, validated_params, span, t_eval_arr)

        sol = solve_ivp(
            fun=self._seiz_derivatives,
            t_span=span,
            y0=y0,
            method="LSODA",
            t_eval=t_eval_arr,
            args=(validated_params,),
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"LSODA ODE integration failed (status={sol.status}): {sol.message}")

        return self._build_result(
            t=sol.t,
            y=sol.y,
            success=bool(sol.success),
            status=int(sol.status),
            message=str(sol.message),
            nfev=getattr(sol, "nfev", 0),
            njev=getattr(sol, "njev", 0),
            nlu=getattr(sol, "nlu", 0),
        )

    @staticmethod
    def _batch_derivatives_numpy(
        y_batch: np.ndarray,
        params_batch: np.ndarray,
    ) -> np.ndarray:
        """Vectorized NumPy SEIZ derivatives for fallback batch RK4 execution."""
        S = y_batch[:, 0]
        E = y_batch[:, 1]
        I = y_batch[:, 2]  # noqa: E741
        Z = y_batch[:, 3]
        inv_n = 1.0 / (S + E + I + Z)

        si_flow = params_batch[:, 0] * S * I * inv_n
        sz_flow = params_batch[:, 3] * S * Z * inv_n
        ei_flow = params_batch[:, 4] * E * I * inv_n
        inc_flow = params_batch[:, 5] * E

        out = np.empty_like(y_batch)
        out[:, 0] = -si_flow - sz_flow
        out[:, 1] = (
            (1.0 - params_batch[:, 1]) * si_flow
            + (1.0 - params_batch[:, 2]) * sz_flow
            - ei_flow
            - inc_flow
        )
        out[:, 2] = params_batch[:, 1] * si_flow + ei_flow + inc_flow
        out[:, 3] = params_batch[:, 2] * sz_flow
        return out

    def solve_batch(
        self,
        state: InitialState | dict[str, int] | Sequence[float] | np.ndarray,
        params_batch: np.ndarray | Sequence[Sequence[float]],
        t_span: tuple[float, float] = (0.0, 100.0),
        num_points: int = 100,
    ) -> np.ndarray:
        """Evaluate [N, 6] parameter sets in batch, returning [N, time_steps, 4]."""
        if num_points < 2:
            raise ValueError("num_points must be at least 2.")

        t0, tf = float(t_span[0]), float(t_span[1])
        if not (math.isfinite(t0) and math.isfinite(tf)) or tf < t0:
            raise ValueError("t_span must be a finite [t0, tf] interval with tf >= t0.")

        params_arr = np.ascontiguousarray(params_batch, dtype=np.float64)
        if params_arr.ndim != 2 or params_arr.shape[1] != 6 or params_arr.shape[0] < 1:
            raise ValueError("params_batch must be a non-empty 2D array of shape (N, 6).")
        if not np.all(np.isfinite(params_arr)):
            raise ValueError("All entries in params_batch must be finite.")
        if np.any(params_arr[:, [0, 3, 4, 5]] < 0.0):
            raise ValueError("Rate parameters (beta, b, rho, epsilon) must be >= 0.0.")
        if np.any(params_arr[:, [1, 2]] < 0.0) or np.any(params_arr[:, [1, 2]] > 1.0):
            raise ValueError("Probability parameters (p, l) must lie within [0.0, 1.0].")

        if isinstance(state, InitialState):
            y0_arr = state.to_array()
        elif isinstance(state, dict):
            y0_arr = InitialState.model_validate(state).to_array()
        else:
            y0_arr = np.ascontiguousarray(state, dtype=np.float64)
            if y0_arr.ndim not in (1, 2):
                raise ValueError("state must have shape (4,) or (N, 4).")

        max_steps = num_points - 1
        dt = (tf - t0) / max_steps

        cpp_active = (
            (self.use_cpp if self.use_cpp is not None else HAS_CPP_CORE)
            and (_cpp_engine is not None)
        )
        if cpp_active:
            assert _cpp_engine is not None
            t_cpp_start = time.perf_counter()
            raw_batch = _cpp_engine.solve_rk4_batch(y0_arr, params_arr, dt, max_steps, t0)
            _record_cpp_duration_ms((time.perf_counter() - t_cpp_start) * 1000.0)
            return np.asarray(raw_batch, dtype=np.float64)

        # Pure-NumPy vectorized RK4 fallback
        n_batch = params_arr.shape[0]
        if y0_arr.ndim == 1:
            y_curr = np.broadcast_to(y0_arr, (n_batch, 4)).copy()
        else:
            y_curr = y0_arr.copy()

        out = np.empty((n_batch, num_points, 4), dtype=np.float64)
        out[:, 0, :] = y_curr
        max_rate = float(np.max(params_arr[:, [0, 3, 4, 5]])) if n_batch > 0 else 0.0
        if dt > 0.0 and max_rate > 0.0:
            sub_steps = max(1, int(math.ceil((max_rate * dt) / 0.2)))
        else:
            sub_steps = 1
        h = dt / sub_steps

        for step in range(1, num_points):
            if dt > 0.0:
                for _ in range(sub_steps):
                    k1 = self._batch_derivatives_numpy(y_curr, params_arr)
                    k2 = self._batch_derivatives_numpy(y_curr + 0.5 * h * k1, params_arr)
                    k3 = self._batch_derivatives_numpy(y_curr + 0.5 * h * k2, params_arr)
                    k4 = self._batch_derivatives_numpy(y_curr + h * k3, params_arr)
                    y_curr += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            out[:, step, :] = y_curr

        return out

    def solve_stochastic(
        self,
        state: InitialState | dict[str, int],
        params: SEIZParameters | dict[str, float],
        t_span: tuple[float, float] | Sequence[float] | np.ndarray,
        t_eval: np.ndarray | Sequence[float] | None = None,
        num_points: int = 500,
        num_simulations: int = 1000,
        seed: int = 42,
        task_id: str | None = None,
    ) -> SimulationResult:
        """Run parallel Gillespie stochastic simulations and return 5th/50th/95th percentiles."""
        return self.solve(
            state=state,
            params=params,
            t_span=t_span,
            t_eval=t_eval,
            num_points=num_points,
            stochastic=True,
            num_simulations=num_simulations,
            seed=seed,
            task_id=task_id,
        )
