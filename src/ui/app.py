"""Interactive Streamlit analyst workbench for the SEIZ Disinformation Simulator."""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Ensure absolute `src.*` imports resolve when executed via `streamlit run src/ui/app.py`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app as fastapi_app  # noqa: E402
from src.api.security import API_KEY_HEADER_NAME, DEFAULT_API_KEY  # noqa: E402
from src.models.schemas import (  # noqa: E402
    CalibrationRequest,
    GlobalScanRequest,
    GlobalSourceName,
    InitialState,
    ParameterSweepRequest,
    SEIZParameters,
    SEIZParamName,
    SimulationRequest,
)
from src.simulation.scenarios import (  # noqa: E402
    SCENARIO_PRESETS,
    build_sample_outbreak_csv,
    compare_intervention_outcomes,
    compute_epidemiological_summary,
    get_preset_by_id,
)

DEFAULT_API_BASE_URL = (
    os.getenv("SEIZ_API_URL")
    or os.getenv("API_BASE_URL")
    or os.getenv("FASTAPI_URL")
    or "http://127.0.0.1:8000"
)

_SWEEP_PARAM_OPTIONS: tuple[SEIZParamName, ...] = ("beta", "p", "l", "b", "rho", "epsilon")
_DEFAULT_POST_SAMPLES = "\n".join(
    [
        "Breaking leaked proof about election_rumor, share this viral video now!",
        "Confirmed cover-up exposed on election_rumor, wake up everyone.",
        "Fact-check: that election_rumor claim was debunked as a fabricated hoax.",
        "False and misleading rumor — independent auditors verified normal operations.",
        "Local news update mentioning election_rumor without taking a stance.",
    ]
)


def compute_viral_inflection_point(
    t: Sequence[float] | np.ndarray,
    I: Sequence[float] | np.ndarray,  # noqa: E741
) -> dict[str, float | int | np.ndarray]:
    """Compute numerical dI/dt and locate the maximum viral velocity inflection point."""
    t_arr = np.asarray(t, dtype=np.float64)
    i_arr = np.asarray(I, dtype=np.float64)

    if t_arr.ndim != 1 or i_arr.ndim != 1:
        raise ValueError("t and I must be 1D sequences.")
    if t_arr.size < 2 or t_arr.shape != i_arr.shape:
        raise ValueError("t and I must have identical lengths >= 2.")
    if not (np.all(np.isfinite(t_arr)) and np.all(np.isfinite(i_arr))):
        raise ValueError("t and I must contain only finite values.")
    if np.any(np.diff(t_arr) <= 0.0):
        raise ValueError("t must be strictly increasing.")

    dI_dt = np.gradient(i_arr, t_arr)
    peak_idx = int(np.argmax(dI_dt))

    return {
        "dI_dt": dI_dt,
        "peak_index": peak_idx,
        "inflection_time": float(t_arr[peak_idx]),
        "max_velocity": float(dI_dt[peak_idx]),
        "infected_at_inflection": float(i_arr[peak_idx]),
    }


def build_seiz_trajectory_figure(
    series: dict[str, Sequence[float] | np.ndarray],
    inflection: dict[str, Any] | None = None,
) -> go.Figure:
    """Render SEIZ median trajectories, 5th–95th percentile bands, and viral inflection marker."""
    t_vals = list(series["t"])
    s_med = series["S_p50"] if series.get("S_p50") is not None else series["S"]
    e_med = series["E_p50"] if series.get("E_p50") is not None else series["E"]
    i_med = series["I_p50"] if series.get("I_p50") is not None else series["I"]
    z_med = series["Z_p50"] if series.get("Z_p50") is not None else series["Z"]

    i_p5 = series["I_p5"] if series.get("I_p5") is not None else i_med
    i_p95 = series["I_p95"] if series.get("I_p95") is not None else i_med
    z_p5 = series["Z_p5"] if series.get("Z_p5") is not None else z_med
    z_p95 = series["Z_p95"] if series.get("Z_p95") is not None else z_med

    if inflection is None:
        inflection = compute_viral_inflection_point(t_vals, i_med)

    t_star = float(inflection["inflection_time"])
    i_star = float(inflection["infected_at_inflection"])
    max_vel = float(inflection["max_velocity"])

    fig = go.Figure()

    ci_bands = [
        (
            "Infected (I) 95th Percentile",
            "Infected (I) 5th–95th Percentile CI",
            i_p95,
            i_p5,
            "rgba(220, 38, 38, 0.2)",
        ),
        (
            "Skeptic (Z) 95th Percentile",
            "Skeptic (Z) 5th–95th Percentile CI",
            z_p95,
            z_p5,
            "rgba(22, 163, 74, 0.2)",
        ),
    ]
    for upper_name, band_name, upper_vals, lower_vals, fill_rgba in ci_bands:
        fig.add_trace(
            go.Scatter(
                x=t_vals,
                y=list(upper_vals),
                mode="lines",
                name=upper_name,
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_vals,
                y=list(lower_vals),
                mode="lines",
                name=band_name,
                fill="tonexty",
                fillcolor=fill_rgba,
                line={"width": 0},
                hoverinfo="skip",
            )
        )

    traces = [
        ("Susceptible (S)", s_med, "#2563eb"),
        ("Exposed (E)", e_med, "#f59e0b"),
        ("Infected (I)", i_med, "#dc2626"),
        ("Skeptic (Z)", z_med, "#16a34a"),
    ]
    for label, y_vals, color in traces:
        fig.add_trace(
            go.Scatter(
                x=t_vals,
                y=list(y_vals),
                mode="lines",
                name=label,
                line={"color": color, "width": 2.5},
            )
        )

    fig.add_vline(
        x=t_star,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text="Maximum Viral Velocity",
        annotation_position="top right",
    )

    fig.add_trace(
        go.Scatter(
            x=[t_star],
            y=[i_star],
            mode="markers",
            name="Maximum Viral Velocity",
            marker={"color": "red", "size": 10, "symbol": "diamond"},
            hovertemplate=(
                "Maximum Viral Velocity<br>"
                "t* = %{x:.3f}<br>"
                "I(t*) = %{y:.2f}<br>"
                f"max dI/dt = {max_vel:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="SEIZ Disinformation Trajectory Simulation",
        xaxis_title="Time (t)",
        yaxis_title="Population",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    return fig


def build_viral_velocity_figure(
    t: Sequence[float] | np.ndarray,
    inflection: dict[str, Any],
) -> go.Figure:
    """Render numerical viral velocity dI/dt over time with the peak velocity highlighted."""
    t_vals = list(t)
    vel_vals = list(np.asarray(inflection["dI_dt"], dtype=np.float64))
    t_star = float(inflection["inflection_time"])
    max_vel = float(inflection["max_velocity"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_vals,
            y=vel_vals,
            mode="lines",
            name="Viral Velocity (dI/dt)",
            fill="tozeroy",
            fillcolor="rgba(220, 38, 38, 0.14)",
            line={"color": "#dc2626", "width": 2.2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[t_star],
            y=[max_vel],
            mode="markers",
            name="Peak Velocity",
            marker={"color": "#991b1b", "size": 10, "symbol": "circle"},
        )
    )
    fig.update_layout(
        title="Narrative Spread Velocity (dI/dt)",
        xaxis_title="Time (t)",
        yaxis_title="New Believers / Time Unit",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def build_intervention_comparison_figure(
    baseline_series: dict[str, Sequence[float] | np.ndarray],
    intervention_series: dict[str, Sequence[float] | np.ndarray],
) -> go.Figure:
    """Overlay Baseline vs Counter-Messaging Intervention Infected & Skeptic curves."""
    t_base = list(baseline_series["t"])
    t_int = list(intervention_series["t"])
    i_base = list(
        baseline_series["I_p50"]
        if baseline_series.get("I_p50") is not None
        else baseline_series["I"]
    )
    z_base = list(
        baseline_series["Z_p50"]
        if baseline_series.get("Z_p50") is not None
        else baseline_series["Z"]
    )
    i_int = list(
        intervention_series["I_p50"]
        if intervention_series.get("I_p50") is not None
        else intervention_series["I"]
    )
    z_int = list(
        intervention_series["Z_p50"]
        if intervention_series.get("Z_p50") is not None
        else intervention_series["Z"]
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_base,
            y=i_base,
            mode="lines",
            name="Baseline Infected (I)",
            line={"color": "#dc2626", "width": 2.5, "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_int,
            y=i_int,
            mode="lines",
            name="Intervention Infected (I)",
            line={"color": "#ef4444", "width": 3.0},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_base,
            y=z_base,
            mode="lines",
            name="Baseline Skeptics (Z)",
            line={"color": "#15803d", "width": 2.0, "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_int,
            y=z_int,
            mode="lines",
            name="Intervention Skeptics (Z)",
            line={"color": "#22c55e", "width": 2.8},
        )
    )
    fig.update_layout(
        title="Baseline vs. Counter-Messaging Intervention",
        xaxis_title="Time (t)",
        yaxis_title="Population",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    return fig


def build_sensitivity_heatmap_figure(
    sweep_data: dict[str, Any],
    metric_key: str = "final_infected_pct",
) -> go.Figure:
    """Render an interactive 2D Plotly heatmap from Eigen3 batched sweep output."""
    metric_labels = {
        "final_infected_pct": ("Final Infected Share (%)", "RdYlGn_r"),
        "max_viral_velocity": ("Max Viral Velocity (dI/dt)", "YlOrRd"),
        "inflection_time": ("Inflection Time (t*)", "Viridis"),
    }
    title_label, colorscale = metric_labels.get(
        metric_key, ("Final Infected Share (%)", "RdYlGn_r")
    )

    fig = go.Figure(
        data=go.Heatmap(
            x=sweep_data["x_values"],
            y=sweep_data["y_values"],
            z=sweep_data[metric_key],
            colorscale=colorscale,
            colorbar={"title": title_label},
        )
    )
    fig.update_layout(
        title=(
            f"Sensitivity Map: {title_label} "
            f"({sweep_data['x_param']} vs {sweep_data['y_param']})"
        ),
        xaxis_title=str(sweep_data["x_param"]),
        yaxis_title=str(sweep_data["y_param"]),
        template="plotly_white",
    )
    return fig


def build_calibration_fit_figure(
    t_observed: Sequence[float],
    i_observed: Sequence[float],
    z_observed: Sequence[float],
    fitted_series: dict[str, Sequence[float]],
) -> go.Figure:
    """Overlay observed (I, Z) points against calibrated SEIZ ODE trajectories."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(t_observed),
            y=list(i_observed),
            mode="markers",
            name="Observed Infected (I)",
            marker={"color": "#b91c1c", "size": 8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(fitted_series["t"]),
            y=list(fitted_series["I"]),
            mode="lines",
            name="Calibrated Infected (I)",
            line={"color": "#dc2626", "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(t_observed),
            y=list(z_observed),
            mode="markers",
            name="Observed Skeptics (Z)",
            marker={"color": "#15803d", "size": 8, "symbol": "square"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(fitted_series["t"]),
            y=list(fitted_series["Z"]),
            mode="lines",
            name="Calibrated Skeptics (Z)",
            line={"color": "#16a34a", "width": 2.5},
        )
    )
    fig.update_layout(
        title="Observed Data vs. Calibrated SEIZ Model Fit",
        xaxis_title="Time (t)",
        yaxis_title="Population",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    return fig


def build_global_spread_map_figure(
    country_stats: Sequence[dict[str, Any]],
    query: str,
) -> go.Figure:
    """Render an ISO-3 world choropleth map of live narrative spread and Infected share."""
    valid_stats = [
        item
        for item in country_stats
        if str(item.get("iso3", "")).upper() not in ("", "UNK")
    ]
    locations = [str(item["iso3"]).upper() for item in valid_stats]
    z_values = [int(item.get("total", 0)) for item in valid_stats]
    hover_text = [
        (
            f"{item.get('country_name', item['iso3'])} ({item['iso3']})<br>"
            f"Total Mentions: {int(item.get('total', 0))}<br>"
            f"Infected: {int(item.get('infected', 0))} "
            f"({float(item.get('infected_share_pct', 0.0)):.1f}%)<br>"
            f"Skeptic: {int(item.get('skeptic', 0))} | Neutral: {int(item.get('neutral', 0))}"
        )
        for item in valid_stats
    ]

    fig = go.Figure(
        data=go.Choropleth(
            locations=locations,
            locationmode="ISO-3",
            z=z_values,
            text=hover_text,
            hoverinfo="text",
            colorscale="Reds",
            autocolorscale=False,
            marker_line_color="rgba(100, 116, 139, 0.5)",
            marker_line_width=0.6,
            colorbar={"title": "Live Mentions"},
        )
    )
    fig.update_layout(
        title=f"Worldwide Narrative Spread Map — '{query}'",
        geo={
            "showframe": False,
            "showcoastlines": True,
            "projection_type": "natural earth",
        },
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def parse_observed_csv(csv_text: str) -> tuple[list[float], list[float], list[float]]:
    """Parse observed (t, I, Z) columns from an uploaded or pasted CSV string."""
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    if reader.fieldnames is None:
        raise ValueError("CSV must contain a header row with t, I, and Z columns.")

    header_map = {name.strip().lower(): name for name in reader.fieldnames if name}
    t_col = header_map.get("t") or header_map.get("time") or header_map.get("t_observed")
    i_col = header_map.get("i") or header_map.get("infected") or header_map.get("i_observed")
    z_col = header_map.get("z") or header_map.get("skeptic") or header_map.get("z_observed")
    if not (t_col and i_col and z_col):
        raise ValueError("CSV must include columns for t (or time), I (or infected), and Z.")

    t_vals: list[float] = []
    i_vals: list[float] = []
    z_vals: list[float] = []
    for row in reader:
        if not row or not row.get(t_col):
            continue
        t_vals.append(float(row[t_col]))
        i_vals.append(float(row[i_col]))
        z_vals.append(float(row[z_col]))

    if len(t_vals) < 2:
        raise ValueError("CSV must contain at least 2 valid numeric rows.")
    return t_vals, i_vals, z_vals


def export_series_to_csv(series: dict[str, Sequence[float]]) -> str:
    """Serialize SEIZ simulation trajectory dictionary into a downloadable CSV."""
    ordered_keys = [
        key
        for key in (
            "t",
            "S",
            "E",
            "I",
            "Z",
            "N",
            "I_p5",
            "I_p50",
            "I_p95",
            "Z_p5",
            "Z_p50",
            "Z_p95",
        )
        if key in series and series[key] is not None
    ]
    n_rows = len(series["t"])
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(ordered_keys)
    for idx in range(n_rows):
        writer.writerow([series[key][idx] for key in ordered_keys])
    return buf.getvalue()


def export_report_to_json(
    series: dict[str, Sequence[float]],
    summary: dict[str, Any],
    parameters: SEIZParameters | None = None,
) -> str:
    """Serialize simulation summary, parameters, and trajectories as pretty JSON."""
    report: dict[str, Any] = {
        "summary": summary,
        "parameters": parameters.model_dump() if parameters is not None else None,
        "series": {k: list(v) for k, v in series.items() if v is not None},
    }
    return json.dumps(report, indent=2)


def _request_with_fallback(
    method: str,
    path: str,
    api_base_url: str = DEFAULT_API_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    timeout: float = 30.0,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    client: Any | None = None,
) -> Any:
    """Send an authenticated API request, falling back to in-process TestClient if offline."""
    headers = {API_KEY_HEADER_NAME: api_key}

    def _dispatch(http_client: Any) -> Any:
        upper = method.upper()
        if upper == "GET":
            return http_client.get(path, params=params, headers=headers)
        if upper == "DELETE":
            return http_client.delete(path, params=params, headers=headers)
        return http_client.post(path, json=json_body, params=params, headers=headers)

    if client is not None:
        resp = _dispatch(client)
    else:
        try:
            with httpx.Client(base_url=api_base_url, timeout=timeout) as http_client:
                resp = _dispatch(http_client)
        except httpx.RequestError:
            with TestClient(fastapi_app) as fallback_client:
                resp = _dispatch(fallback_client)

    if resp.status_code >= 400:
        raise RuntimeError(f"API request {method} {path} failed ({resp.status_code}): {resp.text}")
    return resp.json()


def _submit_and_poll(
    http_client: Any,
    payload: dict[str, Any],
    poll_interval: float,
    timeout: float,
    api_key: str = DEFAULT_API_KEY,
) -> dict[str, list[float]]:
    """Submit a simulation job and poll /api/v1/results/{task_id} until completion."""
    headers = {API_KEY_HEADER_NAME: api_key}
    submit_resp = http_client.post("/api/v1/simulate", json=payload, headers=headers)
    if submit_resp.status_code >= 400:
        raise RuntimeError(
            f"Simulation submission failed ({submit_resp.status_code}): {submit_resp.text}"
        )

    task_id = submit_resp.json()["task_id"]
    deadline = time.monotonic() + timeout

    while True:
        poll_resp = http_client.get(f"/api/v1/results/{task_id}", headers=headers)
        if poll_resp.status_code >= 400:
            raise RuntimeError(
                f"Polling task {task_id} failed ({poll_resp.status_code}): {poll_resp.text}"
            )

        body = poll_resp.json()
        task_status = body["status"]
        if task_status == "completed":
            raw_result = body.get("result")
            if not isinstance(raw_result, dict):
                raise RuntimeError(
                    f"Simulation task {task_id} returned malformed result payload."
                )
            return {
                str(key): [float(val) for val in vals]
                for key, vals in raw_result.items()
                if isinstance(vals, list)
            }
        if task_status == "failed":
            raise RuntimeError(body.get("error") or f"Simulation task {task_id} failed.")

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for simulation task {task_id} after {timeout:.1f}s."
            )
        time.sleep(poll_interval)


def run_simulation_via_api(
    initial_state: InitialState,
    parameters: SEIZParameters,
    t_span: tuple[float, float] = (0.0, 100.0),
    num_points: int = 500,
    stochastic: bool = True,
    num_simulations: int = 1000,
    api_base_url: str = DEFAULT_API_BASE_URL,
    poll_interval: float = 0.05,
    timeout: float = 30.0,
    client: Any | None = None,
    api_key: str = DEFAULT_API_KEY,
) -> dict[str, list[float]]:
    """Call POST /api/v1/simulate and poll GET /api/v1/results/{task_id} until completion."""
    request_model = SimulationRequest(
        initial_state=initial_state,
        parameters=parameters,
        t_span=t_span,
        num_points=num_points,
        stochastic=stochastic,
        num_simulations=num_simulations,
    )
    payload = request_model.model_dump()

    if client is not None:
        return _submit_and_poll(client, payload, poll_interval, timeout, api_key=api_key)

    try:
        with httpx.Client(base_url=api_base_url, timeout=timeout) as http_client:
            return _submit_and_poll(
                http_client, payload, poll_interval, timeout, api_key=api_key
            )
    except httpx.RequestError:
        with TestClient(fastapi_app) as fallback_client:
            return _submit_and_poll(
                fallback_client, payload, poll_interval, timeout, api_key=api_key
            )


def run_calibration_via_api(
    initial_state: InitialState,
    t_observed: Sequence[float],
    i_observed: Sequence[float],
    z_observed: Sequence[float],
    maxiter: int = 300,
    api_base_url: str = DEFAULT_API_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    client: Any | None = None,
) -> dict[str, Any]:
    """Run parameter calibration via POST /api/v1/calibrate and fetch the completed result."""
    req = CalibrationRequest(
        initial_state=initial_state,
        t_observed=[float(v) for v in t_observed],
        I_observed=[float(v) for v in i_observed],
        Z_observed=[float(v) for v in z_observed],
        maxiter=maxiter,
        sync=True,
    )
    submit_body = _request_with_fallback(
        "POST",
        "/api/v1/calibrate",
        api_base_url=api_base_url,
        api_key=api_key,
        json_body=req.model_dump(),
        client=client,
    )
    task_id = str(submit_body["task_id"])
    result_body = _request_with_fallback(
        "GET",
        f"/api/v1/results/{task_id}",
        api_base_url=api_base_url,
        api_key=api_key,
        client=client,
    )
    if result_body.get("status") != "completed" or not isinstance(result_body.get("result"), dict):
        raise RuntimeError(str(result_body.get("error") or "Calibration did not complete."))
    return dict(result_body["result"])


def run_sweep_via_api(
    sweep_request: ParameterSweepRequest,
    api_base_url: str = DEFAULT_API_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    client: Any | None = None,
) -> dict[str, Any]:
    """Execute a 2D Eigen3 batched parameter sweep via POST /api/v1/sweep."""
    body = _request_with_fallback(
        "POST",
        "/api/v1/sweep",
        api_base_url=api_base_url,
        api_key=api_key,
        json_body=sweep_request.model_dump(),
        client=client,
    )
    return dict(body)


def run_global_scan_via_api(
    scan_request: GlobalScanRequest,
    api_base_url: str = DEFAULT_API_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    client: Any | None = None,
) -> dict[str, Any]:
    """Execute a live worldwide multi-source scan via POST /api/v1/global/scan."""
    body = _request_with_fallback(
        "POST",
        "/api/v1/global/scan",
        api_base_url=api_base_url,
        api_key=api_key,
        timeout=45.0,
        json_body=scan_request.model_dump(),
        client=client,
    )
    return dict(body)


def _apply_preset_to_session(preset_id: str) -> None:
    """Load selected scenario preset defaults into Streamlit session state."""
    preset = get_preset_by_id(preset_id)
    if preset is None:
        return
    st.session_state["s0_val"] = preset.initial_state.S0
    st.session_state["e0_val"] = preset.initial_state.E0
    st.session_state["i0_val"] = preset.initial_state.I0
    st.session_state["z0_val"] = preset.initial_state.Z0
    st.session_state["beta_val"] = preset.parameters.beta
    st.session_state["p_val"] = preset.parameters.p
    st.session_state["l_val"] = preset.parameters.l
    st.session_state["b_val"] = preset.parameters.b
    st.session_state["rho_val"] = preset.parameters.rho
    st.session_state["epsilon_val"] = preset.parameters.epsilon
    st.session_state["t_max_val"] = float(preset.t_span[1])


def render_dashboard() -> None:
    """Render the interactive Streamlit SEIZ analyst workbench."""
    st.set_page_config(
        page_title="SEIZ Disinformation Trajectory Simulator",
        layout="wide",
    )
    st.title("SEIZ Disinformation Trajectory Simulator")
    st.caption(
        "Epidemiological Trust & Safety workbench (Susceptible, Exposed, Infected, Skeptic) "
        "powered by a C++17/Eigen3/Gillespie solver, FastAPI, Celery, and PostgreSQL."
    )

    if "s0_val" not in st.session_state:
        st.session_state["s0_val"] = 9900
        st.session_state["e0_val"] = 50
        st.session_state["i0_val"] = 30
        st.session_state["z0_val"] = 20
        st.session_state["beta_val"] = 0.6
        st.session_state["p_val"] = 0.25
        st.session_state["l_val"] = 0.4
        st.session_state["b_val"] = 0.3
        st.session_state["rho_val"] = 0.2
        st.session_state["epsilon_val"] = 0.1
        st.session_state["t_max_val"] = 100.0

    preset_options = {"custom": "Custom Parameters"} | {
        preset.id: f"{preset.name} [{preset.category}]" for preset in SCENARIO_PRESETS
    }
    st.sidebar.header("Scenario Preset")
    selected_preset_id = st.sidebar.selectbox(
        "Disinformation Scenario",
        options=list(preset_options.keys()),
        format_func=lambda key: preset_options[key],
    )
    if (
        selected_preset_id != "custom"
        and st.session_state.get("_active_preset") != selected_preset_id
    ):
        _apply_preset_to_session(selected_preset_id)
        st.session_state["_active_preset"] = selected_preset_id

    active_preset = get_preset_by_id(selected_preset_id)
    if active_preset is not None:
        st.sidebar.caption(active_preset.description)

    st.sidebar.header("Initial Populations")
    s0 = int(
        st.sidebar.number_input(
            "Susceptible (S₀)",
            min_value=0,
            step=10,
            key="s0_val",
            help="Users not yet exposed to the narrative or counter-messaging.",
        )
    )
    e0 = int(
        st.sidebar.number_input(
            "Exposed (E₀)",
            min_value=0,
            step=10,
            key="e0_val",
            help="Users who saw the claim but have not yet adopted/shared it.",
        )
    )
    i0 = int(
        st.sidebar.number_input(
            "Infected (I₀)",
            min_value=0,
            step=10,
            key="i0_val",
            help="Active believers/amplifiers spreading the disinformation narrative.",
        )
    )
    z0 = int(
        st.sidebar.number_input(
            "Skeptic (Z₀)",
            min_value=0,
            step=10,
            key="z0_val",
            help="Fact-checkers and skeptics immune to the narrative who debunk it.",
        )
    )

    st.sidebar.header("SEIZ Parameters")
    beta = float(
        st.sidebar.slider(
            "β (S–I contact rate)",
            0.0,
            5.0,
            step=0.01,
            key="beta_val",
            help="Rate at which susceptible users encounter believers.",
        )
    )
    p = float(
        st.sidebar.slider(
            "p (Prob. S → I)",
            0.0,
            1.0,
            step=0.01,
            key="p_val",
            help="Probability of immediate belief upon contact with a believer.",
        )
    )
    l_prob = float(
        st.sidebar.slider(
            "l (Prob. S → Z)",
            0.0,
            1.0,
            step=0.01,
            key="l_val",
            help="Probability of immediate skepticism upon contact with a debunker.",
        )
    )
    b = float(
        st.sidebar.slider(
            "b (S–Z contact rate)",
            0.0,
            5.0,
            step=0.01,
            key="b_val",
            help="Reach/contact rate of fact-checkers and counter-messaging.",
        )
    )
    rho = float(
        st.sidebar.slider(
            "ρ (E–I adoption rate)",
            0.0,
            5.0,
            step=0.01,
            key="rho_val",
            help="Rate at which exposed users convert after further believer contact.",
        )
    )
    epsilon = float(
        st.sidebar.slider(
            "ε (E → I incubation rate)",
            0.0,
            2.0,
            step=0.01,
            key="epsilon_val",
            help="Spontaneous conversion rate from exposed hesitation to belief.",
        )
    )

    st.sidebar.header("Simulation Horizon")
    stochastic = bool(
        st.sidebar.checkbox("Stochastic Gillespie Simulation (1,000 runs)", value=True)
    )
    t_max = float(
        st.sidebar.number_input("End Time (t_max)", min_value=1.0, step=5.0, key="t_max_val")
    )
    num_points = int(
        st.sidebar.slider("Evaluation Points", min_value=50, max_value=2000, value=500, step=50)
    )

    with st.sidebar.expander("API Connection & Auth", expanded=False):
        api_url = st.text_input("FastAPI Base URL", value=DEFAULT_API_BASE_URL)
        api_key = st.text_input("X-API-Key", value=DEFAULT_API_KEY, type="password")

    run_clicked = st.sidebar.button("Run Simulation", type="primary")

    current_state = InitialState(S0=s0, E0=e0, I0=i0, Z0=z0)
    current_params = SEIZParameters(
        beta=beta,
        p=p,
        l=l_prob,
        b=b,
        rho=rho,
        epsilon=epsilon,
    )

    if run_clicked:
        try:
            with st.spinner("Running SEIZ simulation via FastAPI..."):
                series = run_simulation_via_api(
                    initial_state=current_state,
                    parameters=current_params,
                    t_span=(0.0, t_max),
                    num_points=num_points,
                    stochastic=stochastic,
                    num_simulations=1000,
                    api_base_url=api_url,
                    api_key=api_key,
                )
            st.session_state["simulation_series"] = series
            st.session_state["simulation_N"] = current_state.N
            st.session_state["simulation_params"] = current_params
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

    tab_forecast, tab_global, tab_intervention, tab_calibrate, tab_ops = st.tabs(
        [
            "Trajectory Forecast",
            "Global Narrative Radar",
            "Counter-Messaging & Sensitivity",
            "Calibrate to Data",
            "Classifier & Run History",
        ]
    )

    with tab_forecast:
        if active_preset is not None:
            st.info(
                f"**{active_preset.name}** — {active_preset.description} "
                f"*Recommended play:* {active_preset.recommended_intervention}"
            )

        if "simulation_series" in st.session_state:
            series = st.session_state["simulation_series"]
            sim_params: SEIZParameters = st.session_state.get(
                "simulation_params", current_params
            )
            i_median = series["I_p50"] if series.get("I_p50") is not None else series["I"]
            inflection = compute_viral_inflection_point(series["t"], i_median)
            summary = compute_epidemiological_summary(series, params=sim_params)
            fig = build_seiz_trajectory_figure(series, inflection=inflection)

            col1, col2, col3, col4, col5 = st.columns(5)
            fallback_n = sum(series[k][0] for k in ("S", "E", "I", "Z"))
            total_pop = st.session_state.get("simulation_N", fallback_n)
            col1.metric("Total Population (N)", f"{total_pop:,.0f}")
            col2.metric(
                "Maximum Viral Velocity (max dI/dt)",
                f"{inflection['max_velocity']:.3f}",
            )
            col3.metric("Inflection Timestamp (t*)", f"{inflection['inflection_time']:.2f}")
            col4.metric(
                "Final Infected Share",
                f"{summary['final_infected_pct']:.1f}%",
                delta=f"Skeptics {summary['final_skeptics_pct']:.1f}%",
                delta_color="off",
            )
            rsi_val = summary.get("spreading_ratio_rsi")
            col5.metric(
                "SEIZ Ratio (R_SI)",
                f"{rsi_val:.2f}" if isinstance(rsi_val, float) else "∞",
                delta=str(summary.get("regime_label", "")),
                delta_color="off",
            )

            st.plotly_chart(fig, width="stretch")
            vel_fig = build_viral_velocity_figure(series["t"], inflection)
            st.plotly_chart(vel_fig, width="stretch")

            exp_col1, exp_col2 = st.columns(2)
            exp_col1.download_button(
                "Download Trajectory CSV",
                data=export_series_to_csv(series),
                file_name="seiz_trajectory.csv",
                mime="text/csv",
            )
            exp_col2.download_button(
                "Download JSON Intelligence Report",
                data=export_report_to_json(series, summary, sim_params),
                file_name="seiz_report.json",
                mime="application/json",
            )
        else:
            st.info(
                "Select a disinformation scenario preset or tune SEIZ parameters in the sidebar, "
                "then click **Run Simulation**."
            )

    with tab_global:
        st.subheader("Live Global Narrative Radar (GDELT · Bluesky · Reddit)")
        st.caption(
            "Query public worldwide news and social feeds in real time, classify multilingual "
            "stance (EN/ES/FR/DE/PT/AR + fact-check domains), map ISO-3 country spread, and "
            "fit live SEIZ trajectories."
        )
        gc1, gc2, gc3 = st.columns([2, 2, 1])
        global_query = gc1.text_input("Global Narrative Query", value="vaccine")
        global_sources: list[GlobalSourceName] = gc2.multiselect(
            "Live Feeds",
            options=["gdelt", "bluesky", "reddit"],
            default=["gdelt", "bluesky", "reddit"],
        )
        global_timespan = gc3.selectbox("Window", ["24h", "7d", "14d"], index=1)

        go1, go2, go3 = st.columns(3)
        global_max = go1.slider("Max Items / Source", 5, 80, 30, step=5)
        global_calibrate = go2.checkbox("Auto-calibrate SEIZ & Forecast", value=True)
        global_vel_thresh = go3.number_input(
            "Alert Velocity Threshold (dI/dt)", min_value=0.0, value=5.0, step=0.5
        )

        if st.button("Scan Global Spread"):
            if not global_query.strip() or not global_sources:
                st.warning("Enter a narrative query and select at least one live feed.")
            else:
                try:
                    scan_req = GlobalScanRequest(
                        query=global_query.strip(),
                        sources=global_sources,
                        max_per_source=global_max,
                        timespan=global_timespan,
                        calibrate=global_calibrate,
                        population_size=max(100, int(current_state.N)),
                        alert_velocity_threshold=float(global_vel_thresh),
                    )
                    with st.spinner("Scanning live worldwide feeds and classifying stance..."):
                        scan_res = run_global_scan_via_api(
                            scan_req,
                            api_base_url=api_url,
                            api_key=api_key,
                        )
                    st.session_state["global_scan_result"] = scan_res
                except Exception as exc:
                    st.error(f"Global scan failed: {exc}")

        if "global_scan_result" in st.session_state:
            scan_res = st.session_state["global_scan_result"]
            if scan_res.get("alert_triggered"):
                st.error(f"Viral Alert Triggered: {scan_res.get('alert_message')}")

            stance = scan_res.get("stance_counts", {})
            c_stats = scan_res.get("country_stats", [])
            gm1, gm2, gm3, gm4 = st.columns(4)
            gm1.metric("Total Live Mentions", f"{int(scan_res.get('total_mentions', 0)):,}")
            gm2.metric("Infected (Amplifiers)", f"{int(stance.get('Infected', 0)):,}")
            gm3.metric("Skeptic (Fact-Checks)", f"{int(stance.get('Skeptic', 0)):,}")
            gm4.metric("Countries Reached", f"{len(c_stats):,}")

            st.plotly_chart(
                build_global_spread_map_figure(c_stats, str(scan_res.get("query", ""))),
                width="stretch",
            )

            forecast_s = scan_res.get("forecast_series")
            cal_p_dict = scan_res.get("calibrated_parameters")
            if isinstance(forecast_s, dict) and isinstance(cal_p_dict, dict):
                st.subheader("Live-Calibrated SEIZ Forecast")
                st.plotly_chart(build_seiz_trajectory_figure(forecast_s), width="stretch")
                st.json(cal_p_dict)
                if st.button("Apply Live Calibrated Parameters to Forecast"):
                    st.session_state["beta_val"] = float(cal_p_dict["beta"])
                    st.session_state["p_val"] = float(cal_p_dict["p"])
                    st.session_state["l_val"] = float(cal_p_dict["l"])
                    st.session_state["b_val"] = float(cal_p_dict["b"])
                    st.session_state["rho_val"] = float(cal_p_dict["rho"])
                    st.session_state["epsilon_val"] = float(cal_p_dict["epsilon"])
                    st.session_state["_active_preset"] = "custom"
                    st.success("Live calibrated parameters loaded into sidebar controls.")

            mentions = scan_res.get("mentions", [])
            if mentions:
                st.subheader("Live Geo-Attributed Mentions")
                st.dataframe(mentions[:100], width="stretch")
            else:
                st.info("No live mentions matched this query window across the selected feeds.")

        st.divider()
        st.subheader("Automated Global Watchlist (Celery Beat)")
        wc1, wc2 = st.columns([2, 1])
        watch_name = wc1.text_input("Watchlist Monitor Name", value="Global Health Narrative")
        watch_rsi_thresh = wc2.number_input(
            "Alert R_SI Threshold", min_value=0.0, value=1.0, step=0.1
        )
        wb1, wb2 = st.columns(2)
        if wb1.button("Save Narrative to Watchlist"):
            if not watch_name.strip() or not global_query.strip() or not global_sources:
                st.warning("Provide a monitor name, query, and at least one feed.")
            else:
                try:
                    _request_with_fallback(
                        "POST",
                        "/api/v1/global/narratives",
                        api_base_url=api_url,
                        api_key=api_key,
                        json_body={
                            "name": watch_name.strip(),
                            "query": global_query.strip(),
                            "sources": list(global_sources),
                            "alert_velocity_threshold": float(global_vel_thresh),
                            "alert_rsi_threshold": float(watch_rsi_thresh),
                            "run_initial_scan": False,
                        },
                    )
                    watchlist = _request_with_fallback(
                        "GET",
                        "/api/v1/global/narratives",
                        api_base_url=api_url,
                        api_key=api_key,
                    )
                    st.session_state["global_watchlist"] = watchlist
                    st.success(f"Added '{watch_name.strip()}' to the Celery Beat watchlist.")
                except Exception as exc:
                    st.error(f"Could not save watchlist item: {exc}")

        if wb2.button("Refresh Watchlist"):
            try:
                watchlist = _request_with_fallback(
                    "GET",
                    "/api/v1/global/narratives",
                    api_base_url=api_url,
                    api_key=api_key,
                )
                st.session_state["global_watchlist"] = watchlist
            except Exception as exc:
                st.error(f"Could not load watchlist: {exc}")

        if "global_watchlist" in st.session_state:
            wl_rows = st.session_state["global_watchlist"]
            if wl_rows:
                st.dataframe(wl_rows, width="stretch")
            else:
                st.info("No tracked global narratives yet. Save one above to start monitoring.")

    with tab_intervention:
        st.subheader("Counter-Messaging & Platform Friction What-If")
        st.caption(
            "Model an operational response (e.g., algorithmic forwarding friction reducing β, "
            "or boosted fact-check reach/efficacy raising b and l) against the current baseline."
        )
        c1, c2, c3 = st.columns(3)
        beta_reduction_pct = c1.slider("Platform Friction (% β reduction)", 0, 90, 35, step=5)
        b_boost = c2.slider("Counter-Messaging Reach Boost (+b)", 0.0, 2.0, 0.45, step=0.05)
        l_target = c3.slider(
            "Fact-Check Conversion Target (l)",
            0.0,
            1.0,
            min(1.0, round(l_prob + 0.25, 2)),
            step=0.05,
        )

        if st.button("Compare Intervention vs Baseline"):
            try:
                intervention_params = SEIZParameters(
                    beta=max(0.0, beta * (1.0 - beta_reduction_pct / 100.0)),
                    p=p,
                    l=l_target,
                    b=b + b_boost,
                    rho=rho,
                    epsilon=epsilon,
                )
                with st.spinner("Running baseline & intervention comparison..."):
                    base_series = st.session_state.get("simulation_series")
                    if base_series is None:
                        base_series = run_simulation_via_api(
                            initial_state=current_state,
                            parameters=current_params,
                            t_span=(0.0, t_max),
                            num_points=min(num_points, 300),
                            stochastic=False,
                            api_base_url=api_url,
                            api_key=api_key,
                        )
                    int_series = run_simulation_via_api(
                        initial_state=current_state,
                        parameters=intervention_params,
                        t_span=(0.0, t_max),
                        num_points=min(num_points, 300),
                        stochastic=False,
                        api_base_url=api_url,
                        api_key=api_key,
                    )
                st.session_state["intervention_comparison"] = (base_series, int_series)
            except Exception as exc:
                st.error(f"Intervention comparison failed: {exc}")

        if "intervention_comparison" in st.session_state:
            base_s, int_s = st.session_state["intervention_comparison"]
            impact = compare_intervention_outcomes(base_s, int_s)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Infected Averted", f"{impact['infected_averted']:,.1f}")
            m2.metric("Infection Reduction", f"{impact['infected_averted_pct']:.1f}%")
            m3.metric("Peak Velocity Reduction", f"{impact['peak_velocity_reduction_pct']:.1f}%")
            m4.metric("Inflection Delay (Δt*)", f"{impact['inflection_delay']:+.2f}")
            st.plotly_chart(
                build_intervention_comparison_figure(base_s, int_s),
                width="stretch",
            )

        st.divider()
        st.subheader("Eigen3 Vectorized 2D Parameter Sensitivity Heatmap")
        sc1, sc2, sc3, sc4 = st.columns(4)
        x_param = sc1.selectbox("X-Axis Parameter", _SWEEP_PARAM_OPTIONS, index=0)
        y_default_idx = 2 if x_param != "l" else 3
        y_param = sc2.selectbox("Y-Axis Parameter", _SWEEP_PARAM_OPTIONS, index=y_default_idx)
        grid_res = sc3.slider("Grid Resolution (N × N)", 5, 40, 20, step=5)
        heatmap_metric = sc4.selectbox(
            "Heatmap Metric",
            ["final_infected_pct", "max_viral_velocity", "inflection_time"],
        )

        if st.button("Run Eigen3 Parameter Sweep"):
            if x_param == y_param:
                st.warning("Select two distinct SEIZ parameters for the X and Y axes.")
            else:
                try:
                    x_hi = 1.0 if x_param in ("p", "l") else 1.5
                    y_hi = 1.0 if y_param in ("p", "l") else 1.5
                    sweep_req = ParameterSweepRequest(
                        initial_state=current_state,
                        base_parameters=current_params,
                        x_param=x_param,
                        x_range=(0.05 if x_param not in ("p", "l") else 0.0, x_hi),
                        x_steps=grid_res,
                        y_param=y_param,
                        y_range=(0.05 if y_param not in ("p", "l") else 0.0, y_hi),
                        y_steps=grid_res,
                        t_span=(0.0, t_max),
                        num_points=100,
                    )
                    with st.spinner("Evaluating batched parameter grid on C++ Eigen3 core..."):
                        sweep_res = run_sweep_via_api(
                            sweep_req,
                            api_base_url=api_url,
                            api_key=api_key,
                        )
                    st.session_state["sweep_result"] = sweep_res
                except Exception as exc:
                    st.error(f"Parameter sweep failed: {exc}")

        if "sweep_result" in st.session_state:
            sweep_res = st.session_state["sweep_result"]
            st.caption(
                f"Evaluated {sweep_res['total_combinations']:,} SEIZ ODE trajectories in "
                f"{sweep_res['execution_time_ms']:.2f} ms."
            )
            st.plotly_chart(
                build_sensitivity_heatmap_figure(sweep_res, metric_key=heatmap_metric),
                width="stretch",
            )

    with tab_calibrate:
        st.subheader("Inverse-Problem Calibration (L-BFGS-B)")
        source_mode = st.radio(
            "Observed Data Source",
            ["Sample Outbreak CSV", "Upload / Paste CSV", "Live Ingested Stream"],
            horizontal=True,
        )

        t_obs: list[float] = []
        i_obs: list[float] = []
        z_obs: list[float] = []

        if source_mode == "Sample Outbreak CSV":
            sample_csv = build_sample_outbreak_csv()
            st.download_button(
                "Download Sample CSV Template",
                data=sample_csv,
                file_name="sample_outbreak_observed.csv",
                mime="text/csv",
            )
            t_obs, i_obs, z_obs = parse_observed_csv(sample_csv)
            st.caption(f"Loaded {len(t_obs)} synthetic observed time points (t, I, Z).")
        elif source_mode == "Upload / Paste CSV":
            uploaded = st.file_uploader("Upload CSV with t, I, Z columns", type=["csv"])
            csv_input = (
                uploaded.getvalue().decode("utf-8")
                if uploaded is not None
                else st.text_area(
                    "Or paste CSV rows (t,I,Z)",
                    value=build_sample_outbreak_csv(),
                    height=160,
                )
            )
            try:
                t_obs, i_obs, z_obs = parse_observed_csv(csv_input)
            except Exception as exc:
                st.warning(f"CSV parsing issue: {exc}")
        else:
            live_kw = st.text_input("Tracked Keyword (InfluxDB)", value="disinfo")
            if st.button("Load Live Series"):
                try:
                    live_data = _request_with_fallback(
                        "GET",
                        "/api/v1/data/current",
                        api_base_url=api_url,
                        api_key=api_key,
                        params={"keyword": live_kw, "limit": 60},
                    )
                    st.session_state["live_cal_series"] = live_data
                except Exception as exc:
                    st.error(f"Failed to query live series: {exc}")
            if "live_cal_series" in st.session_state:
                ld = st.session_state["live_cal_series"]
                t_obs = [float(v) for v in ld["t_observed"]]
                i_obs = [float(v) for v in ld["I_observed"]]
                z_obs = [float(v) for v in ld["Z_observed"]]
                st.caption(f"Loaded {len(t_obs)} live minute-level points for '{ld['keyword']}'.")

        if st.button("Calibrate SEIZ Parameters", disabled=len(t_obs) < 2):
            try:
                with st.spinner("Running bounded L-BFGS-B parameter calibration..."):
                    cal_result = run_calibration_via_api(
                        initial_state=current_state,
                        t_observed=t_obs,
                        i_observed=i_obs,
                        z_observed=z_obs,
                        api_base_url=api_url,
                        api_key=api_key,
                    )
                st.session_state["calibration_result"] = {
                    "t_obs": t_obs,
                    "i_obs": i_obs,
                    "z_obs": z_obs,
                    "result": cal_result,
                }
            except Exception as exc:
                st.error(f"Calibration failed: {exc}")

        if "calibration_result" in st.session_state:
            cal_bundle = st.session_state["calibration_result"]
            cal_res = cal_bundle["result"]
            fitted_s = cal_res["fitted_series"]
            cal_params = SEIZParameters(**cal_res["parameters"])
            obs_arr = np.concatenate(
                [
                    np.asarray(cal_bundle["i_obs"], dtype=np.float64),
                    np.asarray(cal_bundle["z_obs"], dtype=np.float64),
                ]
            )
            fit_arr = np.concatenate(
                [
                    np.asarray(fitted_s["I"], dtype=np.float64),
                    np.asarray(fitted_s["Z"], dtype=np.float64),
                ]
            )
            rmse = float(np.sqrt(np.mean((obs_arr - fit_arr) ** 2)))
            ss_tot = float(np.sum((obs_arr - np.mean(obs_arr)) ** 2))
            r_squared = 1.0 - (float(cal_res["sse"]) / ss_tot) if ss_tot > 0.0 else 1.0

            k1, k2, k3 = st.columns(3)
            k1.metric("SSE", f"{cal_res['sse']:.2f}")
            k2.metric("RMSE", f"{rmse:.2f}")
            k3.metric("Goodness of Fit (R²)", f"{r_squared:.4f}")

            st.plotly_chart(
                build_calibration_fit_figure(
                    cal_bundle["t_obs"],
                    cal_bundle["i_obs"],
                    cal_bundle["z_obs"],
                    fitted_s,
                ),
                width="stretch",
            )
            st.json(cal_params.model_dump())

            if st.button("Apply Calibrated Parameters to Forecast"):
                st.session_state["beta_val"] = cal_params.beta
                st.session_state["p_val"] = cal_params.p
                st.session_state["l_val"] = cal_params.l
                st.session_state["b_val"] = cal_params.b
                st.session_state["rho_val"] = cal_params.rho
                st.session_state["epsilon_val"] = cal_params.epsilon
                st.session_state["_active_preset"] = "custom"
                st.success("Calibrated parameters loaded into sidebar controls.")

    with tab_ops:
        st.subheader("Live Social Post Narrative Classifier")
        kw_input = st.text_input("Narrative Keyword Tag", value="election_rumor")
        raw_posts = st.text_area(
            "Paste social posts / headlines (one per line)",
            value=_DEFAULT_POST_SAMPLES,
            height=140,
        )
        persist_flag = st.checkbox("Record minute snapshot to InfluxDB time-series", value=False)
        if st.button("Classify Posts"):
            lines = [line.strip() for line in raw_posts.splitlines() if line.strip()]
            if not lines:
                st.warning("Enter at least one non-empty post line.")
            else:
                try:
                    cls_res = _request_with_fallback(
                        "POST",
                        "/api/v1/ingest/analyze",
                        api_base_url=api_url,
                        api_key=api_key,
                        json_body={
                            "posts": lines,
                            "keyword": kw_input,
                            "persist_to_influx": persist_flag,
                        },
                    )
                    st.session_state["classified_posts"] = cls_res
                except Exception as exc:
                    st.error(f"Classification failed: {exc}")

        if "classified_posts" in st.session_state:
            cls_res = st.session_state["classified_posts"]
            counts = cls_res["counts"]
            p1, p2, p3 = st.columns(3)
            p1.metric("Infected (Narrative Amplifiers)", counts.get("Infected", 0))
            p2.metric("Skeptic (Debunkers)", counts.get("Skeptic", 0))
            p3.metric("Neutral", counts.get("Neutral", 0))
            st.dataframe(cls_res["items"], width="stretch")

        st.divider()
        st.subheader("Persisted Simulation & Calibration Run History")
        if st.button("Refresh Run History"):
            try:
                runs_data = _request_with_fallback(
                    "GET",
                    "/api/v1/runs",
                    api_base_url=api_url,
                    api_key=api_key,
                    params={"limit": 25},
                )
                st.session_state["run_history"] = runs_data.get("runs", [])
            except Exception as exc:
                st.error(f"Could not fetch run history: {exc}")

        if "run_history" in st.session_state:
            rows = st.session_state["run_history"]
            if rows:
                st.dataframe(rows, width="stretch")
            else:
                st.info("No persisted runs recorded yet. Run a simulation or calibration first.")


if __name__ == "__main__":
    render_dashboard()
