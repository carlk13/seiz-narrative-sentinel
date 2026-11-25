"""SEIZ simulation engine and parameter calibration package."""

from src.simulation.calibration import (
    CALIBRATION_DURATION_MS,
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
from src.simulation.scenarios import (
    SCENARIO_PRESETS,
    build_sample_outbreak_csv,
    compare_intervention_outcomes,
    compute_epidemiological_summary,
    compute_spreading_ratio,
    get_preset_by_id,
    get_scenario_presets,
)

__all__ = [
    "CALIBRATION_DURATION_MS",
    "CPP_INTEGRATION_DURATION_MS",
    "CPP_INTEGRATION_SUMMARY_MS",
    "HAS_CPP_CORE",
    "OPTIMIZATION_FAILURES_TOTAL",
    "OptimizationConvergenceError",
    "ParameterEstimator",
    "SCENARIO_PRESETS",
    "SEIZSimulator",
    "SimulationResult",
    "build_sample_outbreak_csv",
    "compare_intervention_outcomes",
    "compute_epidemiological_summary",
    "compute_spreading_ratio",
    "configure_structlog",
    "get_preset_by_id",
    "get_scenario_presets",
]
