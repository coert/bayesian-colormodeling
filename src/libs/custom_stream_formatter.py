import logging
from pathlib import Path
from datetime import datetime
from dateutil.tz.tz import tzlocal


class CustomStreamFormatter(logging.Formatter):
    zone = datetime.now(tzlocal()).tzname()

    def __init__(self, *args, **kwargs):
        if "fmt" in kwargs:
            tzformat = str(kwargs["fmt"])
            if self.zone is not None:
                kwargs["fmt"] = tzformat.replace("%(zone)", self.zone)
            else:
                kwargs["fmt"] = tzformat

        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord):
        full_path = list(Path(record.pathname).parents)[0]

        if record.filename == "__init__.py":
            if "/src/" in str(full_path):
                record.filename = "src" + str(full_path).split("/src")[1]

            else:
                record.filename = full_path.name
        else:
            full_path = list(Path(record.pathname).parents)[0]

            if "/src/" in str(full_path):
                record.filename = (
                    "src" + str(full_path).split("/src")[1] + f"/{record.filename}"
                )

        return super().format(record)
