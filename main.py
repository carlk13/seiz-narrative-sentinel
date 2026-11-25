import structlog

from src.simulation.engine import configure_structlog


def main() -> None:
    """Emit structured readiness log for the CLI entrypoint."""
    configure_structlog()
    logger = structlog.get_logger(__name__)
    logger.info("service_ready", service="seiz-narrative-sentinel")


if __name__ == "__main__":
    main()
