"""Integration tests for the asynchronous FastAPI SEIZ simulation and calibration API."""

from __future__ import annotations

import asyncio
import json
import tomllib
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import numpy as np
import plotly.graph_objects as go
import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy import inspect, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from streamlit.testing.v1 import AppTest

import src.api.routes as api_routes
from alembic import command
from src.api.main import app
from src.api.security import (
    API_KEY_HEADER_NAME,
    DEFAULT_API_KEY,
    reset_rate_limiter,
    resolve_rate_limit_storage_uri,
)
from src.ingestion.poller import (
    InfluxTimeSeriesStore,
    SocialMetricsPoller,
    aggregate_post_counts,
    classify_post,
    floor_to_minute,
)
from src.models.db import AsyncSessionLocal, SimulationRepository, SimulationRun, TrajectoryData
from src.models.schemas import InitialState, SEIZParameters, TaskStatus
from src.simulation.calibration import OptimizationConvergenceError, ParameterEstimator
from src.simulation.engine import SEIZSimulator, configure_structlog
from src.simulation.scenarios import (
    build_sample_outbreak_csv,
    compare_intervention_outcomes,
    compute_epidemiological_summary,
    compute_spreading_ratio,
    get_scenario_presets,
)
from src.ui.app import (
    build_calibration_fit_figure,
    build_intervention_comparison_figure,
    build_seiz_trajectory_figure,
    build_sensitivity_heatmap_figure,
    build_viral_velocity_figure,
    compute_viral_inflection_point,
    export_report_to_json,
    export_series_to_csv,
    parse_observed_csv,
    run_simulation_via_api,
)
from src.workers.celery_app import (
    celery_app,
    get_broker_url,
    get_result_backend,
    run_calibration_task,
    run_simulation_task,
)


@pytest.fixture(autouse=True)
def clear_task_store() -> None:
    """Ensure isolated database rows, TASK_STORE mirror, and rate-limiter buckets."""
    asyncio.run(api_routes.reset_database_state())
    reset_rate_limiter()


@pytest.fixture
def client() -> TestClient:
    """Return an authenticated FastAPI TestClient bound to the SEIZ API application."""
    return TestClient(app, headers={API_KEY_HEADER_NAME: DEFAULT_API_KEY})


class TestHealthEndpoint:
    """Smoke tests for service health endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestSimulationEndpoints:
    """Tests for POST /api/v1/simulate and GET /api/v1/results/{task_id}."""

    def test_simulate_queues_background_task_and_returns_completed_series(
        self,
        client: TestClient,
    ) -> None:
        payload = {
            "initial_state": {"S0": 9900, "E0": 50, "I0": 30, "Z0": 20},
            "parameters": {
                "beta": 0.6,
                "p": 0.25,
                "l": 0.4,
                "b": 0.3,
                "rho": 0.2,
                "epsilon": 0.1,
            },
            "t_span": [0.0, 50.0],
            "num_points": 101,
        }

        submit_resp = client.post("/api/v1/simulate", json=payload)
        assert submit_resp.status_code == 202
        submit_data = submit_resp.json()

        parsed_uuid = uuid.UUID(submit_data["task_id"])
        assert str(parsed_uuid) == submit_data["task_id"]
        assert submit_data["status"] == TaskStatus.PENDING
        assert submit_data["task_type"] == "simulation"

        result_resp = client.get(f"/api/v1/results/{submit_data['task_id']}")
        assert result_resp.status_code == 200
        result_data = result_resp.json()

        assert result_data["task_id"] == submit_data["task_id"]
        assert result_data["status"] == TaskStatus.COMPLETED
        assert result_data["task_type"] == "simulation"
        assert result_data["error"] is None

        series = result_data["result"]
        for key in ("t", "S", "E", "I", "Z", "N"):
            assert len(series[key]) == 101

        totals = (
            np.array(series["S"])
            + np.array(series["E"])
            + np.array(series["I"])
            + np.array(series["Z"])
        )
        np.testing.assert_allclose(totals, 10_000.0, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(series["N"], 10_000.0, rtol=0.0, atol=1e-6)

    def test_task_lifecycle_tracks_pending_processing_and_failed_states(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observed_states: list[TaskStatus] = []
        original_solve = SEIZSimulator.solve

        def spying_solve(
            self: SEIZSimulator,
            *args: object,
            **kwargs: object,
        ):
            # Capture task status while inside the background worker.
            with api_routes._STORE_LOCK:
                for rec in api_routes.TASK_STORE.values():
                    observed_states.append(rec["status"])
            return original_solve(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(SEIZSimulator, "solve", spying_solve)

        payload = {
            "initial_state": {"S0": 100, "E0": 0, "I0": 1, "Z0": 0},
            "parameters": {"beta": 0.4, "p": 0.5, "l": 0.5, "b": 0.2, "rho": 0.1, "epsilon": 0.05},
            "t_span": [0.0, 10.0],
            "num_points": 21,
        }
        submit_resp = client.post("/api/v1/simulate", json=payload)
        assert submit_resp.status_code == 202
        assert submit_resp.json()["status"] == TaskStatus.PENDING
        assert TaskStatus.PROCESSING in observed_states

        # Simulate background failure -> failed status
        def failing_solve(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise RuntimeError("Simulated ODE solver failure")

        monkeypatch.setattr(SEIZSimulator, "solve", failing_solve)
        fail_resp = client.post("/api/v1/simulate", json=payload)
        assert fail_resp.status_code == 202
        fail_id = fail_resp.json()["task_id"]

        result_resp = client.get(f"/api/v1/results/{fail_id}")
        assert result_resp.status_code == 200
        result_data = result_resp.json()
        assert result_data["status"] == TaskStatus.FAILED
        assert "Simulated ODE solver failure" in result_data["error"]

    def test_unknown_task_id_returns_404(self, client: TestClient) -> None:
        resp = client.get(f"/api/v1/results/{uuid.uuid4()}")
        assert resp.status_code == 404


class TestCalibrationEndpoints:
    """Tests for POST /api/v1/calibrate and parameter calibration lifecycle."""

    @staticmethod
    def _build_calibration_payload(sync: bool = False) -> dict[str, object]:
        state = InitialState(S0=950, E0=25, I0=15, Z0=10)
        true_params = SEIZParameters(
            beta=0.65,
            p=0.35,
            l=0.45,
            b=0.25,
            rho=0.2,
            epsilon=0.08,
        )
        t_obs = np.linspace(0.0, 30.0, 16)
        sim = SEIZSimulator().solve(state, true_params, t_span=(0.0, 30.0), t_eval=t_obs)
        return {
            "t_observed": t_obs.tolist(),
            "I_observed": sim.I.tolist(),
            "Z_observed": sim.Z.tolist(),
            "initial_state": {"S0": 950, "E0": 25, "I0": 15, "Z0": 10},
            "initial_guess": [0.45, 0.5, 0.3, 0.15, 0.1, 0.05],
            "maxiter": 200,
            "sync": sync,
        }

    def test_calibrate_async_endpoint_returns_task_and_calibrated_params(
        self,
        client: TestClient,
    ) -> None:
        payload = self._build_calibration_payload(sync=False)
        submit_resp = client.post("/api/v1/calibrate", json=payload)
        assert submit_resp.status_code == 202
        submit_data = submit_resp.json()
        assert uuid.UUID(submit_data["task_id"])
        assert submit_data["status"] == TaskStatus.PENDING
        assert submit_data["task_type"] == "calibration"

        result_resp = client.get(f"/api/v1/results/{submit_data['task_id']}")
        assert result_resp.status_code == 200
        result_data = result_resp.json()
        assert result_data["status"] == TaskStatus.COMPLETED
        assert result_data["task_type"] == "calibration"

        calibrated = SEIZParameters(**result_data["result"]["parameters"])
        assert 0.0 <= calibrated.p <= 1.0
        assert 0.0 <= calibrated.l <= 1.0
        assert result_data["result"]["sse"] < 5.0
        assert len(result_data["result"]["fitted_series"]["t"]) == 16


class TestGlobalExceptionHandlers:
    """Tests for global HTTP 422 handlers on OptimizationConvergenceError and ValidationError."""

    def test_invalid_simulation_and_calibration_payloads_return_formatted_422(
        self,
        client: TestClient,
    ) -> None:
        # Invalid InitialState (negative integer and float) and probability p > 1
        bad_sim_payload = {
            "initial_state": {"S0": -5, "E0": 10.5, "I0": 5, "Z0": 5},
            "parameters": {"beta": -0.2, "p": 1.5, "l": 0.5, "b": 0.1, "rho": 0.1, "epsilon": 0.1},
        }
        sim_resp = client.post("/api/v1/simulate", json=bad_sim_payload)
        assert sim_resp.status_code == 422
        sim_body = sim_resp.json()
        assert sim_body["error"] == "ValidationError"
        assert isinstance(sim_body["detail"], list)
        assert len(sim_body["detail"]) >= 1

        # Mismatched observed array lengths for calibration
        bad_cal_payload = {
            "t_observed": [0.0, 1.0, 2.0],
            "I_observed": [10.0, 12.0],
            "Z_observed": [5.0, 6.0, 7.0],
            "initial_state": {"S0": 100, "E0": 10, "I0": 10, "Z0": 5},
        }
        cal_resp = client.post("/api/v1/calibrate", json=bad_cal_payload)
        assert cal_resp.status_code == 422
        cal_body = cal_resp.json()
        assert cal_body["error"] == "ValidationError"
        assert isinstance(cal_body["detail"], list)

    def test_optimization_convergence_error_returns_formatted_422(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def failing_fit(*args: object, **kwargs: object) -> SEIZParameters:
            del args, kwargs
            raise OptimizationConvergenceError("Singular Hessian inversion failure")

        monkeypatch.setattr("src.api.routes.ParameterEstimator.fit_to_data", failing_fit)

        # Case 1: Direct HTTP 422 via synchronous calibration path
        sync_payload = TestCalibrationEndpoints._build_calibration_payload(sync=True)
        sync_resp = client.post("/api/v1/calibrate", json=sync_payload)
        assert sync_resp.status_code == 422
        sync_body = sync_resp.json()
        assert sync_body["error"] == "OptimizationConvergenceError"
        assert "Singular Hessian inversion failure" in sync_body["detail"]

        # Case 2: Async background calibration stores failed state;
        # raise_on_error=True formats HTTP 422.
        async_payload = TestCalibrationEndpoints._build_calibration_payload(sync=False)
        async_resp = client.post("/api/v1/calibrate", json=async_payload)
        assert async_resp.status_code == 202
        task_id = async_resp.json()["task_id"]

        failed_resp = client.get(f"/api/v1/results/{task_id}")
        assert failed_resp.status_code == 200
        assert failed_resp.json()["status"] == TaskStatus.FAILED

        error_resp = client.get(f"/api/v1/results/{task_id}?raise_on_error=true")
        assert error_resp.status_code == 422
        error_body = error_resp.json()
        assert error_body["error"] == "OptimizationConvergenceError"
        assert "Singular Hessian inversion failure" in error_body["detail"]

    def test_direct_pydantic_validation_error_handler_returns_formatted_422(
        self,
        client: TestClient,
    ) -> None:
        class _StrictPayload(BaseModel):
            count: int

        route_path = "/_test_direct_validation_error"

        @app.get(route_path)
        async def _trigger_validation_error() -> dict[str, str]:
            _StrictPayload(count="not-an-int")  # type: ignore[arg-type]
            return {"status": "unreachable"}

        try:
            resp = client.get(route_path)
            assert resp.status_code == 422
            body = resp.json()
            assert body["error"] == "ValidationError"
            assert isinstance(body["detail"], list)
        finally:
            app.router.routes = [
                r for r in app.router.routes if getattr(r, "path", None) != route_path
            ]


class TestDashboardAndInflectionCalculus:
    """Tests for Streamlit UI, FastAPI polling, Plotly chart, and dI/dt inflection."""

    def test_compute_viral_inflection_point_logistic_and_seiz(self) -> None:
        # Analytical logistic curve: I(t) = L / (1 + exp(-k*(t - t0)))
        # Exact inflection point is t = t0 with max(dI/dt) = k * L / 4.
        t_grid = np.linspace(0.0, 50.0, 501)
        L = 1000.0
        k = 0.4
        t0 = 25.0
        infected_logistic = L / (1.0 + np.exp(-k * (t_grid - t0)))

        inflection = compute_viral_inflection_point(t_grid, infected_logistic)
        assert np.isclose(inflection["inflection_time"], t0, atol=1e-6)
        assert np.isclose(inflection["max_velocity"], k * L / 4.0, rtol=1e-3)
        assert np.isclose(inflection["infected_at_inflection"], L / 2.0, rtol=1e-3)

        # Simulated SEIZ trajectory
        sim = SEIZSimulator().solve(
            InitialState(S0=9900, E0=50, I0=30, Z0=20),
            SEIZParameters(beta=0.6, p=0.25, l=0.4, b=0.3, rho=0.2, epsilon=0.1),
            t_span=(0.0, 100.0),
            num_points=500,
        )
        seiz_inflection = compute_viral_inflection_point(sim.t, sim.I)
        assert 0.0 < float(seiz_inflection["inflection_time"]) < 100.0
        assert float(seiz_inflection["max_velocity"]) > 0.0

    def test_build_seiz_trajectory_figure_contains_curves_and_red_inflection_marker(self) -> None:
        sim = SEIZSimulator().solve_stochastic(
            InitialState(S0=9900, E0=50, I0=30, Z0=20),
            SEIZParameters(beta=0.6, p=0.25, l=0.4, b=0.3, rho=0.2, epsilon=0.1),
            t_span=(0.0, 80.0),
            num_points=201,
            num_simulations=1000,
        )
        series = {
            "t": sim.t.tolist(),
            "S": sim.S.tolist(),
            "E": sim.E.tolist(),
            "I": sim.I.tolist(),
            "Z": sim.Z.tolist(),
            "N": sim.N.tolist(),
            "I_p5": sim.I_p5.tolist(),
            "I_p50": sim.I_p50.tolist(),
            "I_p95": sim.I_p95.tolist(),
            "Z_p5": sim.Z_p5.tolist(),
            "Z_p50": sim.Z_p50.tolist(),
            "Z_p95": sim.Z_p95.tolist(),
        }
        inflection = compute_viral_inflection_point(series["t"], series["I_p50"])
        fig = build_seiz_trajectory_figure(series, inflection=inflection)

        assert isinstance(fig, go.Figure)
        trace_names = [trace.name for trace in fig.data]
        assert "Susceptible (S)" in trace_names
        assert "Exposed (E)" in trace_names
        assert "Infected (I)" in trace_names
        assert "Skeptic (Z)" in trace_names
        assert "Infected (I) 5th–95th Percentile CI" in trace_names
        assert "Skeptic (Z) 5th–95th Percentile CI" in trace_names
        assert "Maximum Viral Velocity" in trace_names

        ci_fill_traces = [trace for trace in fig.data if getattr(trace, "fill", None) == "tonexty"]
        assert len(ci_fill_traces) == 2

        red_vlines = [
            shape
            for shape in fig.layout.shapes
            if shape.line.color == "red"
            and np.isclose(float(shape.x0), float(inflection["inflection_time"]))
            and np.isclose(float(shape.x1), float(inflection["inflection_time"]))
        ]
        assert len(red_vlines) == 1

        annotation_texts = [ann.text for ann in fig.layout.annotations]
        assert "Maximum Viral Velocity" in annotation_texts

    def test_run_simulation_via_api_end_to_end_polling(self, client: TestClient) -> None:
        series = run_simulation_via_api(
            initial_state=InitialState(S0=9900, E0=50, I0=30, Z0=20),
            parameters=SEIZParameters(beta=0.6, p=0.25, l=0.4, b=0.3, rho=0.2, epsilon=0.1),
            t_span=(0.0, 50.0),
            num_points=201,
            stochastic=True,
            num_simulations=1000,
            client=client,
        )

        expected_keys = (
            "t", "S", "E", "I", "Z", "N", "I_p5", "I_p50", "I_p95", "Z_p5", "Z_p50", "Z_p95"
        )
        for key in expected_keys:
            assert len(series[key]) == 201
        np.testing.assert_allclose(series["N"], 10_000.0, rtol=0.0, atol=1e-6)
        assert np.all(np.array(series["I_p5"]) <= np.array(series["I_p50"]) + 1e-12)
        assert np.all(np.array(series["I_p50"]) <= np.array(series["I_p95"]) + 1e-12)
        assert np.all(np.array(series["Z_p5"]) <= np.array(series["Z_p50"]) + 1e-12)
        assert np.all(np.array(series["Z_p50"]) <= np.array(series["Z_p95"]) + 1e-12)

    def test_streamlit_app_ui_renders_and_runs_simulation(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=20)
        at.run()

        assert not at.exception
        assert len(at.sidebar.number_input) >= 4
        assert len(at.sidebar.slider) >= 6
        assert len(at.sidebar.checkbox) >= 1
        assert len(at.sidebar.button) == 1
        assert at.sidebar.button[0].label == "Run Simulation"

        at.sidebar.button[0].click().run()
        assert not at.exception
        assert len(at.error) == 0
        assert len(at.metric) == 5
        assert "simulation_series" in at.session_state
        series = at.session_state["simulation_series"]
        assert "I_p5" in series and "I_p95" in series
        assert "Z_p5" in series and "Z_p95" in series
        assert len(at.get("plotly_chart")) == 2


class TestIngestionDaemonAndDataBridge:
    """Tests for keyword NLP heuristic, InfluxDB v2 store, async poller, and /data/current."""

    def test_deterministic_nlp_keyword_heuristic(self) -> None:
        infected_text = "BREAKING: Leaked proof confirmed! Share this viral truth!"
        skeptic_text = "Fact-check: That misleading claim is a debunked hoax and fake myth."
        neutral_text = "Local weather forecast calls for clear skies tomorrow."
        tie_text = "Breaking rumor was quickly debunked."

        assert classify_post(infected_text) == "Infected"
        assert classify_post(skeptic_text) == "Skeptic"
        assert classify_post(neutral_text) == "Neutral"
        assert classify_post(tie_text) == "Skeptic"
        assert classify_post(infected_text) == classify_post(infected_text)

        counts = aggregate_post_counts(
            [
                {"title": infected_text},
                {"text": skeptic_text},
                {"body": neutral_text},
                "Confirmed cover-up exposed!",
            ]
        )
        assert counts == {"Infected": 2, "Skeptic": 1, "Neutral": 1}

    def test_influx_store_minute_truncation_write_and_flux_query(self) -> None:
        mock_write_api = MagicMock()
        mock_query_api = MagicMock()
        mock_client = MagicMock()
        mock_client.write_api.return_value = mock_write_api
        mock_client.query_api.return_value = mock_query_api

        store = InfluxTimeSeriesStore(
            url="http://127.0.0.1:8086",
            token="test-token",
            org="test-org",
            bucket="test-bucket",
            influx_client=mock_client,
        )

        raw_ts = datetime(2026, 11, 17, 14, 23, 47, 654321, tzinfo=UTC)
        floored = floor_to_minute(raw_ts)
        assert (floored.second, floored.microsecond) == (0, 0)

        point = store.write_minute_counts(
            keyword="vaccine_hoax",
            infected_count=18,
            skeptic_count=7,
            timestamp=raw_ts,
        )
        line_proto = point.to_line_protocol()
        assert "seiz_social_metrics" in line_proto
        assert "keyword=vaccine_hoax" in line_proto
        assert "Infected=18i" in line_proto
        assert "Skeptic=7i" in line_proto
        assert point._time == floored

        mock_write_api.write.assert_called_once()
        write_kwargs = mock_write_api.write.call_args.kwargs
        assert write_kwargs["bucket"] == "test-bucket"
        assert write_kwargs["org"] == "test-org"
        assert write_kwargs["record"] is point

        # Mock FluxTable response to verify Flux parsing
        t0 = datetime(2026, 11, 17, 14, 20, 0, tzinfo=UTC)
        t1 = datetime(2026, 11, 17, 14, 21, 0, tzinfo=UTC)
        rec0 = SimpleNamespace(get_time=lambda: t0, values={"Infected": 12, "Skeptic": 4})
        rec1 = SimpleNamespace(get_time=lambda: t1, values={"Infected": 19, "Skeptic": 9})
        mock_query_api.query.return_value = [SimpleNamespace(records=[rec1, rec0])]

        queried = store.query_series(keyword="vaccine_hoax", limit=10)
        assert queried["keyword"] == "vaccine_hoax"
        assert queried["t_observed"] == [0.0, 1.0]
        assert queried["I_observed"] == [12.0, 19.0]
        assert queried["Z_observed"] == [4.0, 9.0]

    def test_async_poller_polls_mock_rest_api_and_writes_to_store(self) -> None:
        async def _run_poller_scenario() -> tuple[dict[str, int], dict[str, list[object] | str]]:
            mock_posts = {
                "posts": [
                    {"text": "Breaking viral proof on disinfo_topic, share now!"},
                    {"text": "Confirmed leaked truth on disinfo_topic."},
                    {"text": "Fact-check: disinfo_topic claim is a fabricated hoax."},
                    {"text": "Just observing disinfo_topic casually."},
                ]
            }

            def handler(request: httpx.Request) -> httpx.Response:
                assert request.url.params.get("q") == "disinfo_topic"
                return httpx.Response(200, json=mock_posts)

            transport = httpx.MockTransport(handler)
            mock_client = MagicMock()
            mock_client.write_api.return_value = MagicMock()
            mock_client.query_api.return_value = MagicMock(return_value=[])
            mock_client.query_api.return_value.query.return_value = []

            store = InfluxTimeSeriesStore(influx_client=mock_client)
            async with httpx.AsyncClient(transport=transport) as async_client:
                poller = SocialMetricsPoller(
                    api_url="http://mock.social/api/posts",
                    keyword="disinfo_topic",
                    poll_interval_seconds=0.01,
                    store=store,
                    http_client=async_client,
                )
                counts = await poller.poll_once(
                    timestamp=datetime(2026, 11, 17, 15, 0, 35, tzinfo=UTC)
                )
                await poller.run_forever(max_iterations=2)

            return counts, store.query_series(keyword="disinfo_topic", limit=10)

        counts, series = asyncio.run(_run_poller_scenario())
        assert counts == {"Infected": 2, "Skeptic": 1, "Neutral": 1}
        assert len(series["I_observed"]) >= 1
        assert series["I_observed"][-1] == 2.0
        assert series["Z_observed"][-1] == 1.0

    def test_current_data_endpoint_feeds_parameter_estimator(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_client = MagicMock()
        mock_client.write_api.return_value = MagicMock()
        mock_query_api = MagicMock()
        mock_query_api.query.return_value = []
        mock_client.query_api.return_value = mock_query_api

        test_store = InfluxTimeSeriesStore(influx_client=mock_client)
        monkeypatch.setattr(api_routes, "INFLUX_STORE", test_store)

        base_ts = datetime(2026, 11, 17, 12, 0, 0, tzinfo=UTC)
        initial_state = InitialState(S0=950, E0=25, I0=15, Z0=10)
        true_params = SEIZParameters(
            beta=0.65, p=0.35, l=0.45, b=0.25, rho=0.2, epsilon=0.08
        )
        t_grid = np.arange(0.0, 10.0, 1.0)
        sim = SEIZSimulator().solve(
            initial_state, true_params, t_span=(0.0, 9.0), t_eval=t_grid
        )

        for idx, t_min in enumerate(t_grid):
            test_store.write_minute_counts(
                keyword="disinfo",
                infected_count=int(round(sim.I[idx])),
                skeptic_count=int(round(sim.Z[idx])),
                timestamp=base_ts + timedelta(minutes=float(t_min)),
            )

        resp = client.get("/api/v1/data/current?keyword=disinfo&limit=20")
        assert resp.status_code == 200
        body = resp.json()

        assert body["keyword"] == "disinfo"
        assert body["t_observed"] == t_grid.tolist()
        assert len(body["I_observed"]) == 10
        assert len(body["Z_observed"]) == 10
        assert body["I"] == body["I_observed"]
        assert body["Z"] == body["Z_observed"]

        # Confirm arrays plug directly into ParameterEstimator and /api/v1/calibrate.
        fitted = ParameterEstimator().fit_to_data(
            t_observed=body["t_observed"],
            I_observed=body["I_observed"],
            Z_observed=body["Z_observed"],
            initial_state=initial_state,
            maxiter=50,
        )
        assert isinstance(fitted, SEIZParameters)

        cal_resp = client.post(
            "/api/v1/calibrate",
            json={
                "t_observed": body["t_observed"],
                "I_observed": body["I_observed"],
                "Z_observed": body["Z_observed"],
                "initial_state": {"S0": 950, "E0": 25, "I0": 15, "Z0": 10},
                "maxiter": 50,
            },
        )
        assert cal_resp.status_code == 202
        cal_result = client.get(f"/api/v1/results/{cal_resp.json()['task_id']}").json()
        assert cal_result["status"] == TaskStatus.COMPLETED


class TestDockerAndComposeConfiguration:
    """Validation tests for multi-stage Dockerfile and docker-compose.yml."""

    def test_multistage_dockerfile_contains_build_and_runtime_stages(self) -> None:
        root_dir = Path(__file__).resolve().parents[1]
        dockerfile_path = root_dir / "Dockerfile"
        assert dockerfile_path.is_file()

        content = dockerfile_path.read_text(encoding="utf-8")
        assert content.count("FROM python:3.12-slim") == 2
        assert "FROM python:3.12-slim AS builder" in content
        assert "FROM python:3.12-slim AS runtime" in content
        for pkg in ("build-essential", "cmake", "curl"):
            assert pkg in content
        assert "https://astral.sh/uv/install.sh" in content
        assert "src/cpp_core/" in content
        assert "pyproject.toml" in content
        assert "uv sync" in content
        assert "cmake -S . -B build" in content
        assert "cmake --build build" in content
        assert "COPY --from=builder /app/.venv /app/.venv" in content
        assert "COPY --from=builder /app/src /app/src" in content
        assert "COPY src/workers/ ./src/workers/" in content
        assert "COPY --from=builder /app/alembic /app/alembic" in content

    def test_docker_compose_defines_api_ui_bridge_network_and_ports(self) -> None:
        root_dir = Path(__file__).resolve().parents[1]
        compose_path = root_dir / "docker-compose.yml"
        assert compose_path.is_file()

        content = compose_path.read_text(encoding="utf-8")
        assert "postgres:" in content
        assert "postgres:16-alpine" in content
        assert "DATABASE_URL" in content
        assert "redis:" in content
        assert "redis:7-alpine" in content
        assert "celery-worker:" in content
        assert "src.workers.celery_app:celery_app" in content
        assert "CELERY_BROKER_URL" in content
        assert "replicas:" in content

        assert "api:" in content
        assert "uvicorn" in content
        assert "src.api.main:app" in content
        assert '"8000:8000"' in content

        assert "ui:" in content
        assert "streamlit" in content
        assert "src/ui/app.py" in content
        assert '"8501:8501"' in content
        assert "http://api:8000" in content
        assert "SEIZ_API_URL" in content

        assert "networks:" in content
        assert "seiz_network:" in content
        assert "driver: bridge" in content
        assert "volumes:" in content
        assert "seiz_shared_data:" in content


class TestCIWorkflowConfiguration:
    """Validation tests for GitHub Actions CI workflow and static analysis config."""

    def test_github_actions_ci_workflow_spec(self) -> None:
        ci_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
        assert ci_path.exists()
        content = ci_path.read_text(encoding="utf-8")

        assert "pull_request:" in content
        assert "branches:" in content
        assert "- main" in content
        assert "runs-on: ubuntu-latest" in content
        assert "build-essential" in content
        assert "cmake" in content
        assert "astral-sh/setup-uv" in content
        assert "uv sync" in content
        assert "cmake -S . -B build" in content
        assert "cmake --build build" in content
        assert "ruff check ." in content
        assert "mypy src/" in content
        assert "pytest --maxfail=1 --disable-warnings -v" in content

        # Verify CI executes C++ compilation before lint/typecheck/tests.
        assert content.index("uv sync") < content.index("cmake --build build")
        assert content.index("cmake --build build") < content.index("ruff check .")
        assert content.index("ruff check .") < content.index("mypy src/")
        assert content.index("mypy src/") < content.index(
            "pytest --maxfail=1 --disable-warnings -v"
        )

    def test_pyproject_configures_strict_ruff_and_mypy(self) -> None:
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

        dev_deps = data["dependency-groups"]["dev"]
        assert any(dep.startswith("ruff") for dep in dev_deps)
        assert any(dep.startswith("mypy") for dep in dev_deps)

        ruff_cfg = data["tool"]["ruff"]
        assert ruff_cfg["line-length"] == 100
        assert ruff_cfg["target-version"] == "py312"
        assert set(["E", "F", "I", "UP", "B"]).issubset(set(ruff_cfg["lint"]["select"]))

        mypy_cfg = data["tool"]["mypy"]
        assert mypy_cfg["strict"] is True
        assert mypy_cfg["python_version"] == "3.12"
        assert mypy_cfg["disallow_untyped_defs"] is True
        assert "overrides" in mypy_cfg


class TestObservabilityEndpoints:
    """Integration tests for Prometheus /metrics exposition and structlog JSON logs."""

    def test_prometheus_metrics_endpoint_exposes_custom_collectors(
        self,
        client: TestClient,
    ) -> None:
        payload = {
            "initial_state": {"S0": 950, "E0": 25, "I0": 15, "Z0": 10},
            "parameters": {
                "beta": 0.6,
                "p": 0.3,
                "l": 0.4,
                "b": 0.25,
                "rho": 0.2,
                "epsilon": 0.1,
            },
            "t_span": [0.0, 20.0],
            "num_points": 51,
        }
        sim_resp = client.post("/api/v1/simulate", json=payload)
        assert sim_resp.status_code == 202

        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert "text/plain" in metrics_resp.headers["content-type"]

        body = metrics_resp.text
        assert "seiz_cpp_integration_duration_ms_bucket" in body
        assert "seiz_cpp_integration_summary_ms_count" in body
        assert "seiz_optimization_failures_total" in body
        assert "seiz_calibration_duration_ms" in body

    def test_simulate_endpoint_emits_structured_json_logs(
        self,
        client: TestClient,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        configure_structlog()
        capsys.readouterr()

        payload = {
            "initial_state": {"S0": 950, "E0": 25, "I0": 15, "Z0": 10},
            "parameters": {
                "beta": 0.6,
                "p": 0.3,
                "l": 0.4,
                "b": 0.25,
                "rho": 0.2,
                "epsilon": 0.1,
            },
            "t_span": [0.0, 20.0],
            "num_points": 51,
        }
        submit_resp = client.post("/api/v1/simulate", json=payload)
        assert submit_resp.status_code == 202
        task_id = submit_resp.json()["task_id"]

        lines = capsys.readouterr().out.strip().splitlines()
        parsed_events = [json.loads(line) for line in lines if line.strip().startswith("{")]
        matching = [rec for rec in parsed_events if rec.get("task_id") == task_id]
        events_by_name = {rec.get("event"): rec for rec in matching}

        assert "simulation_completed" in events_by_name
        assert "simulation_task_completed" in events_by_name
        for event_name in ("simulation_completed", "simulation_task_completed"):
            rec = events_by_name[event_name]
            assert rec["task_id"] == task_id
            assert isinstance(rec["execution_time_ms"], float)
            assert rec["execution_time_ms"] >= 0.0
            assert rec["solver_status"] == "completed"


class TestSecurityAndRateLimiting:
    """Integration tests for X-API-Key authentication and 5/min slowapi rate limiting."""

    @staticmethod
    def _sample_simulate_payload() -> dict[str, object]:
        return {
            "initial_state": {"S0": 950, "E0": 25, "I0": 15, "Z0": 10},
            "parameters": {
                "beta": 0.6,
                "p": 0.3,
                "l": 0.4,
                "b": 0.25,
                "rho": 0.2,
                "epsilon": 0.1,
            },
            "t_span": [0.0, 10.0],
            "num_points": 21,
        }

    @staticmethod
    def _sample_calibrate_payload() -> dict[str, object]:
        return {
            "initial_state": {"S0": 950, "E0": 25, "I0": 15, "Z0": 10},
            "t_observed": [0.0, 1.0, 2.0, 3.0, 4.0],
            "I_observed": [15.0, 20.0, 27.0, 36.0, 47.0],
            "Z_observed": [10.0, 11.0, 13.0, 15.0, 18.0],
            "maxiter": 2,
            "sync": False,
        }

    def test_missing_or_invalid_api_key_returns_401_unauthorized(self) -> None:
        unauth_client = TestClient(app)
        invalid_headers = {API_KEY_HEADER_NAME: "invalid-api-key-000"}

        endpoints = [
            ("GET", "/health", None),
            ("GET", "/metrics", None),
            ("GET", "/api/v1/results/123", None),
            ("GET", "/api/v1/data/current", None),
            ("POST", "/api/v1/simulate", self._sample_simulate_payload()),
            ("POST", "/api/v1/calibrate", self._sample_calibrate_payload()),
        ]
        for method, path, body in endpoints:
            if method == "GET":
                missing_resp = unauth_client.get(path)
                invalid_resp = unauth_client.get(path, headers=invalid_headers)
            else:
                missing_resp = unauth_client.post(path, json=body)
                invalid_resp = unauth_client.post(path, json=body, headers=invalid_headers)

            assert missing_resp.status_code == 401
            assert invalid_resp.status_code == 401

    def test_simulate_rate_limit_enforces_5_requests_per_minute(
        self,
        client: TestClient,
    ) -> None:
        payload = self._sample_simulate_payload()
        for _ in range(5):
            resp = client.post("/api/v1/simulate", json=payload)
            assert resp.status_code == 202

        limited_resp = client.post("/api/v1/simulate", json=payload)
        assert limited_resp.status_code == 429
        limited_body = limited_resp.json()
        assert limited_body["error"] == "RateLimitExceeded"

    def test_calibrate_rate_limit_enforces_5_requests_per_minute(
        self,
        client: TestClient,
    ) -> None:
        payload = self._sample_calibrate_payload()
        for _ in range(5):
            resp = client.post("/api/v1/calibrate", json=payload)
            assert resp.status_code == 202

        limited_resp = client.post("/api/v1/calibrate", json=payload)
        assert limited_resp.status_code == 429
        limited_body = limited_resp.json()
        assert limited_body["error"] == "RateLimitExceeded"

    def test_redis_storage_uri_fallback_to_memory(self) -> None:
        fallback_uri = resolve_rate_limit_storage_uri("redis://127.0.0.1:63999/0")
        assert fallback_uri == "memory://"
        assert resolve_rate_limit_storage_uri("") == "memory://"


class TestDatabaseModelsAndAlembicMigrations:
    """Tests for async SQLAlchemy ORM persistence, repository transitions, and Alembic."""

    def test_simulation_and_calibration_persist_orm_models_in_db(
        self,
        client: TestClient,
    ) -> None:
        sim_payload = {
            "initial_state": {"S0": 950, "E0": 25, "I0": 15, "Z0": 10},
            "parameters": {
                "beta": 0.6,
                "p": 0.3,
                "l": 0.4,
                "b": 0.25,
                "rho": 0.2,
                "epsilon": 0.1,
            },
            "t_span": [0.0, 20.0],
            "num_points": 41,
        }
        sim_resp = client.post("/api/v1/simulate", json=sim_payload)
        assert sim_resp.status_code == 202
        sim_id = sim_resp.json()["task_id"]

        cal_payload = TestCalibrationEndpoints._build_calibration_payload(sync=False)
        cal_resp = client.post("/api/v1/calibrate", json=cal_payload)
        assert cal_resp.status_code == 202
        cal_id = cal_resp.json()["task_id"]

        async def _verify_db_rows() -> None:
            async with AsyncSessionLocal() as session:
                sim_run = (
                    await session.execute(
                        select(SimulationRun).where(SimulationRun.id == sim_id)
                    )
                ).scalar_one_or_none()
                assert sim_run is not None
                assert sim_run.status == TaskStatus.COMPLETED.value
                assert sim_run.task_type == "simulation"
                assert sim_run.initial_state == sim_payload["initial_state"]
                assert sim_run.parameters == sim_payload["parameters"]
                assert sim_run.created_at is not None
                assert sim_run.updated_at >= sim_run.created_at
                assert sim_run.trajectory is not None
                assert len(sim_run.trajectory.t) == 41
                assert len(sim_run.trajectory.I) == 41
                assert len(sim_run.trajectory.Z) == 41

                cal_run = (
                    await session.execute(
                        select(SimulationRun).where(SimulationRun.id == cal_id)
                    )
                ).scalar_one_or_none()
                cal_traj = (
                    await session.execute(
                        select(TrajectoryData).where(
                            TrajectoryData.simulation_run_id == cal_id
                        )
                    )
                ).scalar_one_or_none()
                assert cal_run is not None
                assert cal_run.status == TaskStatus.COMPLETED.value
                assert cal_run.sse is not None
                assert cal_run.parameters is not None
                assert cal_traj is not None
                assert len(cal_traj.I) == len(cal_traj.t)

        asyncio.run(_verify_db_rows())

    def test_repository_status_transitions_pending_processing_completed_failed(
        self,
        tmp_path: Path,
    ) -> None:
        db_url = f"sqlite+aiosqlite:///{(tmp_path / 'repo_lifecycle.db').as_posix()}"

        async def _exercise_repo() -> None:
            repo_engine = create_async_engine(db_url, poolclass=NullPool, future=True)
            repo = SimulationRepository(target_engine=repo_engine)
            try:
                run_ok_id = str(uuid.uuid4())
                created = await repo.create_run(
                    task_id=run_ok_id,
                    task_type="simulation",
                    status=TaskStatus.PENDING,
                    initial_state={"S0": 100, "E0": 0, "I0": 1, "Z0": 0},
                    parameters={"beta": 0.4},
                )
                assert created.status == TaskStatus.PENDING.value

                processing = await repo.update_status(
                    run_ok_id, status=TaskStatus.PROCESSING
                )
                assert processing is not None
                assert processing.status == TaskStatus.PROCESSING.value

                series = {
                    "t": [0.0, 1.0],
                    "S": [100.0, 99.0],
                    "E": [0.0, 0.5],
                    "I": [1.0, 1.5],
                    "Z": [0.0, 0.0],
                    "N": [101.0, 101.0],
                }
                completed = await repo.update_status(
                    run_ok_id,
                    status=TaskStatus.COMPLETED,
                    series=series,
                )
                assert completed is not None
                assert completed.status == TaskStatus.COMPLETED.value
                assert completed.trajectory is not None
                assert completed.trajectory.to_series_dict() == series

                run_fail_id = str(uuid.uuid4())
                await repo.create_run(
                    task_id=run_fail_id,
                    task_type="calibration",
                    status=TaskStatus.PENDING,
                )
                await repo.update_status(run_fail_id, status=TaskStatus.PROCESSING)
                failed = await repo.update_status(
                    run_fail_id,
                    status=TaskStatus.FAILED,
                    error="Forced failure",
                    error_type="RuntimeError",
                )
                assert failed is not None
                assert failed.status == TaskStatus.FAILED.value
                assert failed.error == "Forced failure"
                assert failed.error_type == "RuntimeError"
            finally:
                await repo_engine.dispose()

        asyncio.run(_exercise_repo())

    def test_alembic_async_upgrade_and_downgrade(self, tmp_path: Path) -> None:
        db_url = f"sqlite+aiosqlite:///{(tmp_path / 'alembic_async.db').as_posix()}"
        ini_path = Path(__file__).resolve().parents[1] / "alembic.ini"
        cfg = Config(str(ini_path))
        cfg.set_main_option("sqlalchemy.url", db_url)

        async def _get_tables_and_columns() -> tuple[set[str], set[str], set[str]]:
            check_engine = create_async_engine(db_url, poolclass=NullPool, future=True)
            try:
                async with check_engine.connect() as conn:
                    tables = set(
                        await conn.run_sync(lambda c: inspect(c).get_table_names())
                    )
                    run_cols = (
                        set(
                            await conn.run_sync(
                                lambda c: [
                                    col["name"]
                                    for col in inspect(c).get_columns("simulation_runs")
                                ]
                            )
                        )
                        if "simulation_runs" in tables
                        else set()
                    )
                    traj_cols = (
                        set(
                            await conn.run_sync(
                                lambda c: [
                                    col["name"]
                                    for col in inspect(c).get_columns("trajectory_data")
                                ]
                            )
                        )
                        if "trajectory_data" in tables
                        else set()
                    )
                    return tables, run_cols, traj_cols
            finally:
                await check_engine.dispose()

        command.upgrade(cfg, "head")
        tables_up, run_cols, traj_cols = asyncio.run(_get_tables_and_columns())
        assert {"alembic_version", "simulation_runs", "trajectory_data"}.issubset(
            tables_up
        )
        assert {
            "id",
            "task_type",
            "status",
            "initial_state",
            "parameters",
            "sse",
            "created_at",
            "updated_at",
        }.issubset(run_cols)
        assert {
            "id",
            "simulation_run_id",
            "t",
            "S",
            "E",
            "I",
            "Z",
            "N",
            "percentiles",
        }.issubset(traj_cols)

        command.downgrade(cfg, "base")
        tables_down, _, _ = asyncio.run(_get_tables_and_columns())
        assert "simulation_runs" not in tables_down
        assert "trajectory_data" not in tables_down


class TestCeleryWorkerAndBrokerIntegration:
    """Tests for Celery app config, @celery.task registration, and task.delay dispatch."""

    def test_celery_app_configuration_and_task_registration(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("CELERY_BROKER_URL", "redis://redis-host:6379/2")
        monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://redis-host:6379/3")
        assert get_broker_url() == "redis://redis-host:6379/2"
        assert get_result_backend() == "redis://redis-host:6379/3"

        assert "src.workers.celery_app.run_simulation_task" in celery_app.tasks
        assert "src.workers.celery_app.run_calibration_task" in celery_app.tasks
        assert callable(run_simulation_task.delay)
        assert callable(run_calibration_task.delay)
        assert celery_app.conf.task_serializer == "json"

    def test_simulate_endpoint_uses_celery_task_delay(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mocked_task_id = f"celery-spy-{uuid.uuid4()}"
        dispatched: list[dict[str, object]] = []

        class _SpyTask:
            def delay(self, payload_dict: dict[str, object]) -> SimpleNamespace:
                dispatched.append(payload_dict)
                return SimpleNamespace(id=mocked_task_id, state="PENDING")

        monkeypatch.setattr(api_routes, "run_simulation_task", _SpyTask())

        payload = {
            "initial_state": {"S0": 950, "E0": 25, "I0": 15, "Z0": 10},
            "parameters": {
                "beta": 0.6,
                "p": 0.3,
                "l": 0.4,
                "b": 0.25,
                "rho": 0.2,
                "epsilon": 0.1,
            },
            "t_span": [0.0, 20.0],
            "num_points": 31,
        }
        resp = client.post("/api/v1/simulate", json=payload)
        assert resp.status_code == 202
        body = resp.json()
        assert body["task_id"] == mocked_task_id
        assert body["status"] == TaskStatus.PENDING
        assert len(dispatched) == 1
        assert dispatched[0]["num_points"] == 31

        status_resp = client.get(f"/api/v1/results/{mocked_task_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == TaskStatus.PENDING

    def test_celery_tasks_execute_solve_and_fit_to_data_directly(self) -> None:
        sim_payload = {
            "initial_state": {"S0": 950, "E0": 25, "I0": 15, "Z0": 10},
            "parameters": {
                "beta": 0.6,
                "p": 0.3,
                "l": 0.4,
                "b": 0.25,
                "rho": 0.2,
                "epsilon": 0.1,
            },
            "t_span": [0.0, 10.0],
            "num_points": 21,
        }
        sim_async = run_simulation_task.delay(sim_payload)
        assert uuid.UUID(sim_async.id)
        assert sim_async.successful() is True
        sim_series = sim_async.result
        assert len(sim_series["t"]) == 21
        np.testing.assert_allclose(sim_series["N"], 1000.0, rtol=0.0, atol=1e-6)

        cal_payload = TestCalibrationEndpoints._build_calibration_payload(sync=False)
        cal_async = run_calibration_task.delay(cal_payload)
        assert uuid.UUID(cal_async.id)
        assert cal_async.successful() is True
        cal_result = cal_async.result
        assert "parameters" in cal_result
        assert cal_result["sse"] < 5.0

        saved_cal = asyncio.run(api_routes.SIMULATION_REPO.get_run_dict(cal_async.id))
        assert saved_cal is not None
        assert saved_cal["status"] == TaskStatus.COMPLETED
        assert saved_cal["result"]["sse"] < 5.0


class TestAnalystWorkbenchAndEndpoints:
    """Tests for analyst presets, 2D Eigen3 sweeps, run history, classification, and exports."""

    def test_presets_endpoint_and_epidemiological_analytics(
        self,
        client: TestClient,
    ) -> None:
        resp = client.get("/api/v1/presets")
        assert resp.status_code == 200
        presets = resp.json()
        assert len(presets) == len(get_scenario_presets()) >= 5
        assert any(item["id"] == "breaking_election_rumor" for item in presets)

        base_params = SEIZParameters(
            beta=0.85, p=0.45, l=0.25, b=0.20, rho=0.35, epsilon=0.12
        )
        r_si, regime = compute_spreading_ratio(base_params)
        assert r_si is not None and r_si > 0.0
        assert isinstance(regime, str) and len(regime) > 0

        state = InitialState(S0=9900, E0=40, I0=45, Z0=15)
        int_params = SEIZParameters(
            beta=0.45, p=0.45, l=0.70, b=0.75, rho=0.35, epsilon=0.12
        )
        base_sim = SEIZSimulator().solve(state, base_params, t_span=(0.0, 60.0), num_points=121)
        int_sim = SEIZSimulator().solve(state, int_params, t_span=(0.0, 60.0), num_points=121)
        base_dict = {
            "t": base_sim.t,
            "S": base_sim.S,
            "E": base_sim.E,
            "I": base_sim.I,
            "Z": base_sim.Z,
            "N": base_sim.N,
        }
        int_dict = {
            "t": int_sim.t,
            "S": int_sim.S,
            "E": int_sim.E,
            "I": int_sim.I,
            "Z": int_sim.Z,
            "N": int_sim.N,
        }

        summary = compute_epidemiological_summary(base_dict, params=base_params)
        assert summary["total_population"] == pytest.approx(10_000.0)
        assert summary["final_infected_pct"] > 0.0
        assert summary["max_viral_velocity"] > 0.0

        impact = compare_intervention_outcomes(base_dict, int_dict)
        assert impact["infected_averted"] > 0.0
        assert impact["peak_velocity_reduction_pct"] > 0.0
        assert impact["additional_skeptics"] > 0.0

    def test_parameter_sweep_endpoint_and_heatmap_builder(
        self,
        client: TestClient,
    ) -> None:
        payload = {
            "initial_state": {"S0": 9900, "E0": 40, "I0": 45, "Z0": 15},
            "base_parameters": {
                "beta": 0.6,
                "p": 0.3,
                "l": 0.4,
                "b": 0.35,
                "rho": 0.2,
                "epsilon": 0.1,
            },
            "x_param": "beta",
            "x_range": [0.1, 1.2],
            "x_steps": 10,
            "y_param": "l",
            "y_range": [0.0, 1.0],
            "y_steps": 10,
            "t_span": [0.0, 60.0],
            "num_points": 60,
        }
        resp = client.post("/api/v1/sweep", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_combinations"] == 100
        assert len(body["x_values"]) == 10
        assert len(body["y_values"]) == 10
        assert len(body["final_infected_pct"]) == 10
        assert len(body["final_infected_pct"][0]) == 10
        assert body["execution_time_ms"] >= 0.0

        fig = build_sensitivity_heatmap_figure(body, metric_key="final_infected_pct")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

        bad_payload = dict(payload, y_param="beta")
        bad_resp = client.post("/api/v1/sweep", json=bad_payload)
        assert bad_resp.status_code == 422

    def test_run_history_endpoint_lists_persisted_runs(
        self,
        client: TestClient,
    ) -> None:
        sim_payload = {
            "initial_state": {"S0": 950, "E0": 25, "I0": 15, "Z0": 10},
            "parameters": {
                "beta": 0.6,
                "p": 0.3,
                "l": 0.4,
                "b": 0.25,
                "rho": 0.2,
                "epsilon": 0.1,
            },
            "t_span": [0.0, 10.0],
            "num_points": 21,
        }
        sub = client.post("/api/v1/simulate", json=sim_payload)
        assert sub.status_code == 202
        task_id = sub.json()["task_id"]

        hist_resp = client.get("/api/v1/runs?limit=10&task_type=simulation")
        assert hist_resp.status_code == 200
        hist = hist_resp.json()
        assert hist["total"] >= 1
        assert any(run["task_id"] == task_id for run in hist["runs"])

    def test_ingest_analyze_endpoint_classifies_and_persists_posts(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_client = MagicMock()
        mock_client.write_api.return_value = MagicMock()
        mock_query_api = MagicMock()
        mock_query_api.query.return_value = []
        mock_client.query_api.return_value = mock_query_api

        test_store = InfluxTimeSeriesStore(influx_client=mock_client)
        monkeypatch.setattr(api_routes, "INFLUX_STORE", test_store)

        resp = client.post(
            "/api/v1/ingest/analyze",
            json={
                "keyword": "rumor_x",
                "persist_to_influx": True,
                "posts": [
                    "Breaking leaked proof on rumor_x, share this viral thread!",
                    "Fact-check: rumor_x was debunked as a fabricated hoax.",
                    "General commentary on rumor_x.",
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["persisted"] is True
        assert body["counts"] == {"Infected": 1, "Skeptic": 1, "Neutral": 1}
        assert [item["label"] for item in body["items"]] == [
            "Infected",
            "Skeptic",
            "Neutral",
        ]

        current = client.get("/api/v1/data/current?keyword=rumor_x&limit=10")
        assert current.status_code == 200
        assert current.json()["I_observed"][-1] == 1.0
        assert current.json()["Z_observed"][-1] == 1.0

    def test_csv_parsing_export_and_analyst_figures(self) -> None:
        sample_csv = build_sample_outbreak_csv()
        t_obs, i_obs, z_obs = parse_observed_csv(sample_csv)
        assert len(t_obs) == 16
        assert len(i_obs) == 16
        assert len(z_obs) == 16

        state = InitialState(S0=950, E0=25, I0=15, Z0=10)
        params = SEIZParameters(beta=0.6, p=0.3, l=0.4, b=0.25, rho=0.2, epsilon=0.1)
        sim = SEIZSimulator().solve(state, params, t_span=(0.0, 20.0), num_points=41)
        series = {
            "t": sim.t.tolist(),
            "S": sim.S.tolist(),
            "E": sim.E.tolist(),
            "I": sim.I.tolist(),
            "Z": sim.Z.tolist(),
            "N": sim.N.tolist(),
        }
        csv_out = export_series_to_csv(series)
        assert csv_out.startswith("t,S,E,I,Z,N")
        assert len(csv_out.strip().splitlines()) == 42

        summary = compute_epidemiological_summary(series, params=params)
        json_out = export_report_to_json(series, summary, params)
        parsed = json.loads(json_out)
        assert parsed["parameters"]["beta"] == pytest.approx(0.6)
        assert len(parsed["series"]["t"]) == 41

        inflection = compute_viral_inflection_point(series["t"], series["I"])
        assert isinstance(build_viral_velocity_figure(series["t"], inflection), go.Figure)
        assert isinstance(build_intervention_comparison_figure(series, series), go.Figure)
        assert isinstance(
            build_calibration_fit_figure(t_obs, i_obs, z_obs, series), go.Figure
        )
