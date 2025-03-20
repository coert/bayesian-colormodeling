import logging

from pathlib import Path

from .libs.custom_stream_formatter import CustomStreamFormatter  # type: ignore


def setup_logger(
    my_logger: logging.Logger,
    level: int = logging.INFO,
    format: str = "[%(process)d] %(levelname)-1s [%(filename)s:%(lineno)d] %(message)s",
    log_file: Path | None = None,
):
    my_logger.handlers = []
    my_logger.setLevel(level)

    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(
        CustomStreamFormatter(
            fmt=("[%(asctime)s.%(msecs)03d %(zone)] " + format),
            datefmt="%d-%m-%Y:%H:%M:%S",
        )
    )
    my_logger.addHandler(handler_stream)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            CustomStreamFormatter(
                fmt=("[%(asctime)s.%(msecs)03d %(zone)] " + format),
                datefmt="%d-%m-%Y:%H:%M:%S",
            )
        )
        my_logger.addHandler(file_handler)
