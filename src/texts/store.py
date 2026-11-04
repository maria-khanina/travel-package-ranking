"""Reads and appends generated texts stored as JSON lines."""

from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from src.texts.models import TextRecord

ENCODING = "utf-8"
REJECTED_SUFFIX = "_rejected.jsonl"


def read_records(path: Path) -> list[TextRecord]:
    """Reads every stored record of a job output file."""
    records: list[TextRecord] = []
    if not path.exists():
        return records
    with open(path, encoding=ENCODING) as handle:
        for line in handle:
            records.append(TextRecord.model_validate_json(line))
    return records


def read_text_map(path: Path) -> dict[int, str]:
    """Maps item ids to their successfully generated texts."""
    texts: dict[int, str] = {}
    for record in read_records(path):
        if record.text:
            texts[record.id] = record.text
    return texts


def open_for_append(path: Path) -> TextIO:
    """Opens a job output file for appending, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding=ENCODING)


def append_record(handle: TextIO, record: TextRecord) -> None:
    """Appends one record to an open output file and flushes it."""
    handle.write(record.model_dump_json() + "\n")
    handle.flush()


def choose_records(
    records: list[TextRecord],
    accepts: Callable[[TextRecord], bool],
) -> dict[int, TextRecord]:
    """Keeps for every item its latest acceptable record, else an empty one."""
    chosen: dict[int, TextRecord] = {}
    for record in records:
        empty = record.model_copy(update={"text": None})
        candidate = record if accepts(record) else empty
        current = chosen.get(record.id)
        if candidate.text is not None or current is None or current.text is None:
            chosen[record.id] = candidate
    return chosen


def save_rejected(
    path: Path,
    records: list[TextRecord],
    accepts: Callable[[TextRecord], bool],
) -> None:
    """Keeps the texts that no longer pass validation in a side file."""
    rejected = path.with_name(f"{path.stem}{REJECTED_SUFFIX}")
    with open(rejected, "a", encoding=ENCODING) as handle:
        for record in records:
            if record.text is not None and not accepts(record):
                append_record(handle, record)


def compact_records(path: Path, accepts: Callable[[TextRecord], bool]) -> None:
    """Rewrites a job file with one record per item, sorted by item id."""
    if not path.exists():
        return
    records = read_records(path)
    save_rejected(path, records, accepts)
    chosen = choose_records(records, accepts)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", encoding=ENCODING) as handle:
        for identifier in sorted(chosen):
            append_record(handle, chosen[identifier])
    temporary.replace(path)
