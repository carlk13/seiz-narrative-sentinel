"""Curated disinformation scenario presets and epidemiological analytics for SEIZ models."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.models.schemas import InitialState, ScenarioPreset, SEIZParameters
from src.simulation.engine import SEIZSimulator

SCENARIO_PRESETS: tuple[ScenarioPreset, ...] = (
    ScenarioPreset(
        id="breaking_election_rumor",
        name="Breaking Election Rumor (High Virality)",
        category="Civic Integrity",
        description=(
            "Fast-moving fabricated claim about polling disruptions with high direct sharing "
            "(high β and p) and limited early debunking visibility."
        ),
        recommended_intervention=(
            "Deploy rapid verified counter-messaging (increase b >= 0.55 and l >= 0.60) "
            "before the viral inflection point to blunt peak velocity."
        ),
        initial_state=InitialState(S0=9900, E0=40, I0=45, Z0=15),
        parameters=SEIZParameters(
            beta=0.85,
            p=0.45,
            l=0.25,
            b=0.20,
            rho=0.35,
            epsilon=0.12,
        ),
        t_span=(0.0, 60.0),
    ),
    ScenarioPreset(
        id="health_misinfo_slow_burn",
        name="Health Misinformation (Slow Incubation)",
        category="Public Health",
        description=(
            "Persistent medical claim where most exposed readers hesitate in compartment E "
            "before adopting via repeated peer exposure (low p, moderate ρ and ε)."
        ),
        recommended_intervention=(
            "Pre-bunk exposed audiences and boost trusted clinical debunking (increase l) "
            "during the extended incubation window."
        ),
        initial_state=InitialState(S0=9850, E0=90, I0=35, Z0=25),
        parameters=SEIZParameters(
            beta=0.55,
            p=0.12,
            l=0.30,
            b=0.22,
            rho=0.18,
            epsilon=0.06,
        ),
        t_span=(0.0, 100.0),
    ),
    ScenarioPreset(
        id="rapid_fact_check_response",
        name="Rapid Fact-Check Counter-Campaign",
        category="Intervention Benchmark",
        description=(
            "Coordinated early response by fact-checkers and platform context labels "
            "(elevated skeptic pool Z0, high b and l) suppressing narrative takeover."
        ),
        recommended_intervention=(
            "Maintain counter-messaging reach (b) and label clarity (l) through the early "
            "window to lock in majority skeptic immunity."
        ),
        initial_state=InitialState(S0=9750, E0=50, I0=40, Z0=160),
        parameters=SEIZParameters(
            beta=0.55,
            p=0.22,
            l=0.72,
            b=0.68,
            rho=0.15,
            epsilon=0.05,
        ),
        t_span=(0.0, 80.0),
    ),
    ScenarioPreset(
        id="polarized_echo_chamber",
        name="Polarized Echo Chamber",
        category="Information Operations",
        description=(
            "Competing believer and skeptic communities with strong internal reinforcement "
            "(balanced β/b and elevated ρ), splitting the network into entrenched factions."
        ),
        recommended_intervention=(
            "Introduce cross-cutting friction (reduce β and ρ) and trusted bridge messengers "
            "to shift undecided exposed users before entrenchment."
        ),
        initial_state=InitialState(S0=9600, E0=80, I0=160, Z0=160),
        parameters=SEIZParameters(
            beta=0.70,
            p=0.30,
            l=0.55,
            b=0.55,
            rho=0.40,
            epsilon=0.08,
        ),
        t_span=(0.0, 80.0),
    ),
    ScenarioPreset(
        id="coordinated_bot_amplification",
        name="Coordinated Bot Amplification",
        category="Inauthentic Behavior",
        description=(
            "Artificial amplification spike combining aggressive contact rate (β=1.25) with "
            "high direct adoption and rapid exposed-to-infected conversion."
        ),
        recommended_intervention=(
            "Throttle automated forwarding / algorithmic promotion (cut β by >= 50%) "
            "simultaneously with counter-narrative placement."
        ),
        initial_state=InitialState(S0=9800, E0=70, I0=110, Z0=20),
        parameters=SEIZParameters(
            beta=1.25,
            p=0.55,
            l=0.18,
            b=0.15,
            rho=0.50,
            epsilon=0.15,
        ),
        t_span=(0.0, 50.0),
    ),
)


def get_scenario_presets() -> list[ScenarioPreset]:
    """Return all built-in disinformation scenario presets."""
    return list(SCENARIO_PRESETS)


def get_preset_by_id(preset_id: str) -> ScenarioPreset | None:
    """Look up a scenario preset by its unique identifier."""
    for preset in SCENARIO_PRESETS:
        if preset.id == preset_id:
            return preset
    return None


def compute_spreading_ratio(params: SEIZParameters) -> tuple[float | None, str]:
    """Compute the Bettencourt/Jin SEIZ spreading ratio R_SI and classify the regime.

    R_SI = ((1 - p) * beta + (1 - l) * b) / (rho + epsilon)
    Values < 1 indicate rapid/direct viral adoption; values > 1 indicate an
    incubation-dominated narrative where users spend time in the Exposed state.
    """
    denom = params.rho + params.epsilon
    if denom <= 0.0:
        return None, "Static Incubation (ρ + ε = 0)"

    r_si = ((1.0 - params.p) * params.beta + (1.0 - params.l) * params.b) / denom
    if r_si < 1.0:
        regime = "Direct Viral Spread (Fast Adoption)"
    elif r_si <= 3.0:
        regime = "Mixed Diffusion & Incubation"
    else:
        regime = "Incubation-Dominated (High Hesitation)"
    return float(r_si), regime


def compute_epidemiological_summary(
    series: Mapping[str, Sequence[float] | np.ndarray],
    params: SEIZParameters | None = None,
) -> dict[str, Any]:
    """Compute analyst-ready KPIs from a simulated SEIZ trajectory."""
    t_arr = np.asarray(series["t"], dtype=np.float64)
    i_src = series["I_p50"] if series.get("I_p50") is not None else series["I"]
    z_src = series["Z_p50"] if series.get("Z_p50") is not None else series["Z"]
    i_arr = np.asarray(i_src, dtype=np.float64)
    z_arr = np.asarray(z_src, dtype=np.float64)

    if "N" in series and len(series["N"]) > 0:
        total_n = float(np.asarray(series["N"], dtype=np.float64)[0])
    else:
        total_n = float(
            sum(float(np.asarray(series[k], dtype=np.float64)[0]) for k in ("S", "E", "I", "Z"))
        )

    di_dt = np.gradient(i_arr, t_arr)
    peak_vel_idx = int(np.argmax(di_dt))
    peak_i_idx = int(np.argmax(i_arr))

    safe_n = total_n if total_n > 0.0 else 1.0
    summary: dict[str, Any] = {
        "total_population": total_n,
        "peak_infected": float(i_arr[peak_i_idx]),
        "peak_infected_pct": float(100.0 * i_arr[peak_i_idx] / safe_n),
        "peak_infected_time": float(t_arr[peak_i_idx]),
        "final_infected": float(i_arr[-1]),
        "final_infected_pct": float(100.0 * i_arr[-1] / safe_n),
        "final_skeptics": float(z_arr[-1]),
        "final_skeptics_pct": float(100.0 * z_arr[-1] / safe_n),
        "inflection_time": float(t_arr[peak_vel_idx]),
        "max_viral_velocity": float(di_dt[peak_vel_idx]),
        "infected_at_inflection": float(i_arr[peak_vel_idx]),
    }
    if params is not None:
        r_si, regime = compute_spreading_ratio(params)
        summary["spreading_ratio_rsi"] = r_si
        summary["regime_label"] = regime
    return summary


def compare_intervention_outcomes(
    baseline_series: dict[str, Sequence[float] | np.ndarray],
    intervention_series: dict[str, Sequence[float] | np.ndarray],
) -> dict[str, float]:
    """Quantify counter-messaging / friction impact comparing intervention vs baseline."""
    base_kpi = compute_epidemiological_summary(baseline_series)
    int_kpi = compute_epidemiological_summary(intervention_series)

    base_final_i = float(base_kpi["final_infected"])
    int_final_i = float(int_kpi["final_infected"])
    infected_averted = base_final_i - int_final_i
    infected_averted_pct = (
        (100.0 * infected_averted / base_final_i) if base_final_i > 0.0 else 0.0
    )

    base_vel = float(base_kpi["max_viral_velocity"])
    int_vel = float(int_kpi["max_viral_velocity"])
    velocity_reduction_pct = (
        (100.0 * (base_vel - int_vel) / base_vel) if base_vel > 0.0 else 0.0
    )

    return {
        "infected_averted": infected_averted,
        "infected_averted_pct": infected_averted_pct,
        "peak_velocity_reduction_pct": velocity_reduction_pct,
        "inflection_delay": float(int_kpi["inflection_time"]) - float(base_kpi["inflection_time"]),
        "additional_skeptics": float(int_kpi["final_skeptics"]) - float(base_kpi["final_skeptics"]),
    }


def build_sample_outbreak_csv() -> str:
    """Generate a realistic sample CSV (t, I, Z) for one-click calibration testing."""
    preset = SCENARIO_PRESETS[0]
    t_grid = np.linspace(0.0, 30.0, 16)
    sim = SEIZSimulator().solve(
        state=preset.initial_state,
        params=preset.parameters,
        t_span=(0.0, 30.0),
        t_eval=t_grid,
        _log_simulation=False,
    )
    rng = np.random.default_rng(7)
    max_n = float(preset.initial_state.N)
    i_noisy = np.clip(sim.I + rng.normal(0.0, 8.0, size=t_grid.size), 0.0, max_n)
    z_noisy = np.clip(sim.Z + rng.normal(0.0, 4.0, size=t_grid.size), 0.0, max_n)
    i_noisy[0] = float(preset.initial_state.I0)
    z_noisy[0] = float(preset.initial_state.Z0)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["t", "I", "Z"])
    for t_val, i_val, z_val in zip(t_grid, i_noisy, z_noisy, strict=True):
        writer.writerow([f"{t_val:.2f}", f"{i_val:.2f}", f"{z_val:.2f}"])
    return buf.getvalue()
