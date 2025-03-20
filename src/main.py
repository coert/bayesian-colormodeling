import click
import logging

from pathlib import Path

from .utils import setup_logger

logger = logging.getLogger(__package__)


def init_logging(log_level: str, log_file_path: Path | None = None) -> None:
    if log_file_path is not None:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

    match log_level.lower():
        case "debug":
            level = logging.DEBUG
        case "info":
            level = logging.INFO
        case "warning":
            level = logging.WARNING
        case "error":
            level = logging.ERROR
        case _:
            level = logging.INFO

    click.echo(f"Logging level: {log_level}")

    setup_logger(
        my_logger=logger,
        level=level,
        format="[%(process)d] %(levelname)-1s [%(filename)s:%(lineno)d] %(message)s",
        log_file=log_file_path,
    )


@click.group()
@click.option("--debug", help="debug level", default="debug", type=str)
@click.option("--log_file_path", help="log file path", default=None, type=Path)
def cli(debug: str, log_file_path: Path | None):
    init_logging(debug, log_file_path)


def main():
    cli()


if __name__ == "__main__":
    main()
