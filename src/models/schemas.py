"""Strict Pydantic schemas for the SEIZ disinformation epidemiological model."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)


class InitialState(BaseModel):
    """Initial compartment counts $(S_0, E_0, I_0, Z_0)$ for the SEIZ model."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    S0: StrictInt = Field(
        ...,
        ge=0,
        validation_alias=AliasChoices("S0", "S_0"),
        description="Initial susceptible population",
    )
    E0: StrictInt = Field(
        ...,
        ge=0,
        validation_alias=AliasChoices("E0", "E_0"),
        description="Initial exposed population",
    )
    I0: StrictInt = Field(
        ...,
        ge=0,
        validation_alias=AliasChoices("I0", "I_0"),
        description="Initial infected population",
    )
    Z0: StrictInt = Field(
        ...,
        ge=0,
        validation_alias=AliasChoices("Z0", "Z_0"),
        description="Initial skeptic population",
    )

    @field_validator("S0", "E0", "I0", "Z0", mode="before")
    @classmethod
    def _validate_non_negative_int(cls, value: object) -> int:
        """Reject booleans, floats, and negative values; require strict integers >= 0."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "Initial population counts must be strict integers, not booleans or floats."
            )
        if value < 0:
            raise ValueError("Initial population counts must be non-negative (>= 0).")
        return value

    @model_validator(mode="after")
    def _validate_positive_total_population(self) -> InitialState:
        """Ensure total initial population N = S0 + E0 + I0 + Z0 is strictly positive."""
        if self.N <= 0:
            raise ValueError(
                "Total initial population N = S0 + E0 + I0 + Z0 must be strictly positive (> 0)."
            )
        return self

    @property
    def S_0(self) -> int:
        """Alias for initial susceptible population $S_0$."""
        return self.S0

    @property
    def E_0(self) -> int:
        """Alias for initial exposed population $E_0$."""
        return self.E0

    @property
    def I_0(self) -> int:
        """Alias for initial infected population $I_0$."""
        return self.I0

    @property
    def Z_0(self) -> int:
        """Alias for initial skeptic population $Z_0$."""
        return self.Z0

    @property
    def N(self) -> int:
        """Total conserved population $N = S_0 + E_0 + I_0 + Z_0$."""
        return self.S0 + self.E0 + self.I0 + self.Z0

    def to_array(self) -> np.ndarray:
        """Convert initial compartment counts to a float64 NumPy state vector [S0, E0, I0, Z0]."""
        return np.array([self.S0, self.E0, self.I0, self.Z0], dtype=np.float64)


class SEIZParameters(BaseModel):
    """Rate and probability parameters $(\\beta, p, l, b, \\rho, \\epsilon)$ for the SEIZ ODEs."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)

    beta: float = Field(..., ge=0.0, description="S-I contact rate")
    p: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of direct S -> I transition given contact with I",
    )
    l: float = Field(  # noqa: E741
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of direct S -> Z transition given contact with Z",
    )
    b: float = Field(..., ge=0.0, description="S-Z contact rate")
    rho: float = Field(..., ge=0.0, description="E-I contact adoption rate")
    epsilon: float = Field(..., ge=0.0, description="Spontaneous E -> I incubation rate")

    @field_validator("beta", "p", "l", "b", "rho", "epsilon", mode="before")
    @classmethod
    def _validate_finite_real(cls, value: object) -> object:
        """Reject booleans and non-finite numeric values (NaN, Inf)."""
        if isinstance(value, bool):
            raise ValueError("SEIZ parameters must be finite real numbers, not booleans.")
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(
                "SEIZ parameters must be finite real numbers (NaN and Inf are not allowed)."
            )
        return value

    @field_validator("beta", "b", "rho", "epsilon")
    @classmethod
    def _validate_non_negative_rate(cls, value: float) -> float:
        """Enforce that rate parameters are finite and >= 0."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Rate parameters must be finite and >= 0.0.")
        return value

    @field_validator("p", "l")
    @classmethod
    def _validate_probability(cls, value: float) -> float:
        """Enforce that transition probabilities p and l lie within [0, 1]."""
        if not math.isfinite(value) or not (0.0 <= value <= 1.0):
            raise ValueError("Probability parameters p and l must lie within [0.0, 1.0].")
        return value


class TaskStatus(StrEnum):
    """Lifecycle states for asynchronous background simulation and calibration tasks."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SimulationRequest(BaseModel):
    """Request payload for triggering an asynchronous SEIZ forward simulation."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)

    initial_state: InitialState = Field(
        ...,
        validation_alias=AliasChoices("initial_state", "state"),
        description="Initial SEIZ compartment counts",
    )
    parameters: SEIZParameters = Field(
        ...,
        validation_alias=AliasChoices("parameters", "params"),
        description="SEIZ ODE rate and probability parameters",
    )
    t_span: tuple[float, float] = Field(
        default=(0.0, 100.0),
        description="Integration time interval [t0, tf]",
    )
    num_points: StrictInt = Field(
        default=500,
        ge=2,
        le=100_000,
        description="Number of uniformly spaced evaluation points over t_span",
    )
    stochastic: bool = Field(
        default=False,
        description=(
            "If true, execute parallel Gillespie stochastic simulations instead of "
            "deterministic ODEs"
        ),
    )
    num_simulations: StrictInt = Field(
        default=1000,
        ge=1,
        le=50_000,
        description="Number of Monte Carlo trajectories when stochastic=True",
    )

    @field_validator("num_points", "num_simulations", mode="before")
    @classmethod
    def _reject_bool_int_fields(cls, value: object) -> object:
        """Reject boolean values for strict integer fields."""
        if isinstance(value, bool):
            raise ValueError("Integer simulation configuration fields must not be booleans.")
        return value

    @field_validator("t_span", mode="before")
    @classmethod
    def _reject_bool_in_t_span(cls, value: object) -> object:
        """Reject boolean values inside t_span."""
        if isinstance(value, (tuple, list)) and any(isinstance(item, bool) for item in value):
            raise ValueError("t_span endpoints must be finite real numbers, not booleans.")
        return value

    @field_validator("t_span")
    @classmethod
    def _validate_t_span(cls, value: tuple[float, float]) -> tuple[float, float]:
        """Enforce finite t0, tf with tf >= t0."""
        t0, tf = value
        if not (math.isfinite(t0) and math.isfinite(tf)):
            raise ValueError("t_span endpoints must be finite numbers.")
        if tf < t0:
            raise ValueError("t_span end time must be >= start time.")
        return (float(t0), float(tf))


class CalibrationRequest(BaseModel):
    """Request payload for calibrating SEIZ parameters to observed (I, Z) time series."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)

    t_observed: list[float] = Field(..., min_length=2, description="Strictly increasing timestamps")
    I_observed: list[float] = Field(..., min_length=2, description="Observed infected counts")
    Z_observed: list[float] = Field(..., min_length=2, description="Observed skeptic counts")
    initial_state: InitialState = Field(
        ...,
        validation_alias=AliasChoices("initial_state", "state"),
        description="Known initial compartment counts",
    )
    initial_guess: list[float] | SEIZParameters | None = Field(
        default=None,
        description="Optional 6-element [beta, p, l, b, rho, epsilon] guess or SEIZParameters",
    )
    maxiter: StrictInt = Field(
        default=500,
        ge=1,
        le=10_000,
        description="Maximum L-BFGS-B optimization iterations",
    )
    sync: bool = Field(
        default=False,
        description="Optional flag to run calibration synchronously",
    )

    @field_validator("maxiter", mode="before")
    @classmethod
    def _reject_bool_maxiter(cls, value: object) -> object:
        """Reject boolean values for strict integer field maxiter."""
        if isinstance(value, bool):
            raise ValueError("maxiter must be an integer, not a boolean.")
        return value

    @model_validator(mode="after")
    def _validate_observed_series(self) -> CalibrationRequest:
        """Validate matching lengths, finiteness, and monotonic timestamps."""
        n = len(self.t_observed)
        if n < 2:
            raise ValueError("Observed time series must contain at least 2 points.")
        if len(self.I_observed) != n or len(self.Z_observed) != n:
            raise ValueError(
                "t_observed, I_observed, and Z_observed must have identical lengths."
            )

        for series in (self.t_observed, self.I_observed, self.Z_observed):
            if not all(math.isfinite(val) for val in series):
                raise ValueError("Observed arrays must contain only finite real numbers.")

        if any(self.t_observed[i + 1] <= self.t_observed[i] for i in range(n - 1)):
            raise ValueError("t_observed must be strictly increasing.")

        if isinstance(self.initial_guess, list):
            if len(self.initial_guess) != 6:
                raise ValueError(
                    "initial_guess list must have length 6: [beta, p, l, b, rho, epsilon]."
                )
            if not all(math.isfinite(val) for val in self.initial_guess):
                raise ValueError("All initial_guess values must be finite real numbers.")

        return self


class SimulationSeriesPayload(BaseModel):
    """JSON-serializable compartment trajectory time series and percentile bands."""

    t: list[float]
    S: list[float]
    E: list[float]
    I: list[float]  # noqa: E741
    Z: list[float]
    N: list[float]
    S_p5: list[float] | None = None
    S_p50: list[float] | None = None
    S_p95: list[float] | None = None
    E_p5: list[float] | None = None
    E_p50: list[float] | None = None
    E_p95: list[float] | None = None
    I_p5: list[float] | None = None
    I_p50: list[float] | None = None
    I_p95: list[float] | None = None
    Z_p5: list[float] | None = None
    Z_p50: list[float] | None = None
    Z_p95: list[float] | None = None


class CalibrationResultPayload(BaseModel):
    """Calibrated SEIZ parameters and goodness-of-fit trajectory summary."""

    parameters: SEIZParameters
    sse: float
    fitted_series: SimulationSeriesPayload


class TaskSubmissionResponse(BaseModel):
    """Immediate acknowledgement returned upon submitting an asynchronous API task."""

    task_id: str
    status: TaskStatus
    task_type: Literal["simulation", "calibration"]


class TaskStatusResponse(BaseModel):
    """Status and optional payload returned by GET /api/v1/results/{task_id}."""

    task_id: str
    status: TaskStatus
    task_type: Literal["simulation", "calibration"]
    result: SimulationSeriesPayload | CalibrationResultPayload | dict[str, Any] | None = None
    error: str | None = None


class ErrorResponse(BaseModel):
    """Standardized HTTP 422 error payload."""

    error: str
    detail: Any


class CurrentDataResponse(BaseModel):
    """Formatted social time-series response ready for ParameterEstimator ingestion."""

    keyword: str
    timestamps: list[str]
    t_observed: list[float]
    I_observed: list[float]
    Z_observed: list[float]
    I: list[float]  # noqa: E741
    Z: list[float]


SEIZParamName = Literal["beta", "p", "l", "b", "rho", "epsilon"]


class ScenarioPreset(BaseModel):
    """Curated real-world disinformation scenario preset for analyst exploration."""

    id: str
    name: str
    category: str
    description: str
    recommended_intervention: str
    initial_state: InitialState
    parameters: SEIZParameters
    t_span: tuple[float, float] = (0.0, 100.0)


class ParameterSweepRequest(BaseModel):
    """Request payload for Eigen3-vectorized 2D SEIZ parameter sensitivity sweeps."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)

    initial_state: InitialState = Field(
        ...,
        validation_alias=AliasChoices("initial_state", "state"),
        description="Initial SEIZ compartment counts",
    )
    base_parameters: SEIZParameters = Field(
        ...,
        validation_alias=AliasChoices("base_parameters", "parameters", "params"),
        description="Baseline SEIZ parameters held constant for non-swept axes",
    )
    x_param: SEIZParamName = Field(default="beta", description="Parameter varied along X axis")
    x_range: tuple[float, float] = Field(default=(0.1, 1.5), description="[min, max] for x_param")
    x_steps: StrictInt = Field(default=20, ge=2, le=50, description="Grid points along X axis")
    y_param: SEIZParamName = Field(default="l", description="Parameter varied along Y axis")
    y_range: tuple[float, float] = Field(default=(0.0, 1.0), description="[min, max] for y_param")
    y_steps: StrictInt = Field(default=20, ge=2, le=50, description="Grid points along Y axis")
    t_span: tuple[float, float] = Field(default=(0.0, 100.0), description="Simulation horizon")
    num_points: StrictInt = Field(default=100, ge=2, le=500, description="Evaluation time steps")

    @field_validator("x_steps", "y_steps", "num_points", mode="before")
    @classmethod
    def _reject_bool_sweep_ints(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Grid dimensions and num_points must be integers, not booleans.")
        return value

    @model_validator(mode="after")
    def _validate_sweep_bounds(self) -> ParameterSweepRequest:
        if self.x_param == self.y_param:
            raise ValueError("x_param and y_param must be distinct SEIZ parameters.")
        if self.x_steps * self.y_steps > 2500:
            raise ValueError("Total sweep combinations (x_steps * y_steps) must be <= 2500.")
        t0, tf = self.t_span
        if not (math.isfinite(t0) and math.isfinite(tf)) or tf <= t0:
            raise ValueError("t_span must satisfy finite t0 < tf.")

        for name, bounds in ((self.x_param, self.x_range), (self.y_param, self.y_range)):
            low, high = bounds
            if not (math.isfinite(low) and math.isfinite(high)) or high < low:
                raise ValueError(f"Range for {name} must satisfy finite low <= high.")
            if low < 0.0:
                raise ValueError(f"Lower bound for {name} must be >= 0.0.")
            if name in ("p", "l") and high > 1.0:
                raise ValueError(f"Upper bound for probability parameter {name} must be <= 1.0.")
        return self


class ParameterSweepResponse(BaseModel):
    """Compact 2D sensitivity grid produced by the Eigen3 batched RK4 solver."""

    x_param: SEIZParamName
    y_param: SEIZParamName
    x_values: list[float]
    y_values: list[float]
    final_infected_pct: list[list[float]]
    max_viral_velocity: list[list[float]]
    inflection_time: list[list[float]]
    total_combinations: int
    execution_time_ms: float


class RunSummaryItem(BaseModel):
    """Summary metadata for a persisted simulation or calibration run."""

    task_id: str
    task_type: str
    status: TaskStatus
    initial_state: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None
    sse: float | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class RunHistoryResponse(BaseModel):
    """Paginated list of recent persisted SimulationRun records."""

    total: int
    runs: list[RunSummaryItem]


class PostClassificationRequest(BaseModel):
    """Batch of raw social posts to classify into Infected, Skeptic, or Neutral."""

    model_config = ConfigDict(extra="forbid")

    posts: list[str] = Field(..., min_length=1, max_length=500)
    keyword: str = Field(default="disinfo", min_length=1, max_length=100)
    persist_to_influx: bool = Field(
        default=False,
        description="If true, write the aggregated minute counts to InfluxTimeSeriesStore",
    )


class PostClassificationItem(BaseModel):
    """Classification result for a single social post."""

    text: str
    label: Literal["Infected", "Skeptic", "Neutral"]


class PostClassificationResponse(BaseModel):
    """Aggregated counts and per-post classifications for analyst inspection."""

    keyword: str
    counts: dict[str, int]
    items: list[PostClassificationItem]
    persisted: bool
