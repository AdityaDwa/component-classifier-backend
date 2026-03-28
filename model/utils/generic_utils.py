from pathlib import Path
from datetime import datetime, timezone

import pytz


def get_current_datetime(tz_str: str = "Asia/Kathmandu") -> str:
    """
    Function that returns the current date and time in the specified timezone.
    Defaults to UTC if no timezone is provided.

    Args:
        tz_str (str): Timezone string (e.g., 'Asia/Kathmandu'). Defaults to 'UTC'.

    Returns:
        str: Current date and time in the format "YYYY-MM-DD HH:MM:SS.mmmmmm", where the microseconds are represented
            by the 'mmmmmm' part.
    """
    tz = pytz.timezone(tz_str)
    current_time = datetime.now(timezone.utc).astimezone(tz)
    return current_time.strftime("%Y-%m-%d %H:%M:%S.%f")


def remove_file(file_path: Path) -> None:
    """
    Removes the file at the given path if it exists. Prints a confirmation message if the file is removed, or a
    warning if the file does not exist.

    Args:
        file_path (Path): Path object representing the file to be removed.

    Returns:
        None
    """
    if file_path.exists():
        file_path.unlink()
        print(f"File removed: {file_path}")
    else:
        print("File does not exist")
