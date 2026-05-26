"""TIMESTAMPTZ normalization helpers.

Milvus stores TIMESTAMPTZ as UTC Unix microseconds on the wire.  The
engine accepts user-friendly ISO 8601 strings and timezone-aware
``datetime`` objects, then normalizes them to the same canonical integer.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from milvus_lite.exceptions import SchemaValidationError


_UTC = timezone.utc
_EPOCH = datetime(1970, 1, 1, tzinfo=_UTC)
_INTERVAL_RE = re.compile(
    r"^P"
    r"(?:(?P<weeks>\d+(?:\.\d+)?)W)?"
    r"(?:(?P<days>\d+(?:\.\d+)?)D)?"
    r"(?:T"
    r"(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
    r")?$",
    re.IGNORECASE,
)


def parse_timestamptz(value: Any, default_timezone: str | None = None) -> int:
    """Normalize a TIMESTAMPTZ value to UTC Unix microseconds.

    Accepted inputs:
    - int: treated as already-normalized UTC microseconds
    - timezone-aware datetime
    - ISO 8601 string with an offset or ``Z`` suffix

    Naive strings/datetimes are rejected unless ``default_timezone`` is
    provided by collection/request timezone properties.
    """
    if isinstance(value, bool):
        raise SchemaValidationError("TIMESTAMPTZ value must not be bool")
    if isinstance(value, int):
        return value
    if isinstance(value, datetime):
        return datetime_to_unix_micros(_ensure_aware(value, default_timezone))
    if isinstance(value, str):
        return datetime_to_unix_micros(_parse_datetime_string(value, default_timezone))
    raise SchemaValidationError(
        f"TIMESTAMPTZ value must be ISO string, aware datetime, or int microseconds; "
        f"got {type(value).__name__}"
    )


def validate_timezone_name(value: Any) -> str:
    """Validate and normalize a collection/request timezone property."""
    if not isinstance(value, str) or not value:
        raise SchemaValidationError("timezone property must be a non-empty string")
    try:
        ZoneInfo(value)
    except ZoneInfoNotFoundError as e:
        raise SchemaValidationError(f"unknown timezone {value!r}") from e
    return value


def datetime_to_unix_micros(value: datetime) -> int:
    """Convert an aware datetime to UTC Unix microseconds."""
    value = value.astimezone(_UTC)
    delta = value - _EPOCH
    return (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )


def micros_to_utc_datetime(value: Any) -> datetime | None:
    """Convert UTC Unix microseconds or an aware datetime to UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_aware(value, None).astimezone(_UTC)
    if isinstance(value, bool) or not isinstance(value, int):
        value = parse_timestamptz(value)
    return _EPOCH + timedelta(microseconds=int(value))


def micros_to_iso_z(value: Any) -> str | None:
    """Render a TIMESTAMPTZ value as an ISO 8601 UTC string with ``Z``."""
    dt = micros_to_utc_datetime(value)
    if dt is None:
        return None
    text = dt.isoformat().replace("+00:00", "Z")
    if text.endswith(".000000Z"):
        text = text.replace(".000000Z", "Z")
    return text


def extract_time_fields(
    value: Any,
    fields: list[str],
    timezone_name: str = "UTC",
) -> list[int] | None:
    """Extract TIMESTAMPTZ components in the requested timezone."""
    dt = micros_to_utc_datetime(value)
    if dt is None:
        return None
    tz = ZoneInfo(validate_timezone_name(timezone_name))
    local = dt.astimezone(tz)
    out: list[int] = []
    for field in fields:
        key = field.casefold()
        if key == "year":
            out.append(local.year)
        elif key == "month":
            out.append(local.month)
        elif key == "day":
            out.append(local.day)
        elif key == "hour":
            out.append(local.hour)
        elif key == "minute":
            out.append(local.minute)
        elif key == "second":
            out.append(local.second)
        elif key == "microsecond":
            out.append(local.microsecond)
        else:
            raise SchemaValidationError(
                f"unsupported time_fields component {field!r}; supported: "
                "year, month, day, hour, minute, second, microsecond"
            )
    return out


def parse_interval_micros(value: str) -> int:
    """Parse a small ISO 8601 duration subset into microseconds.

    Supported units: weeks, days, hours, minutes, seconds. Calendar
    years/months are deliberately rejected because their duration depends
    on a reference date.
    """
    if not isinstance(value, str):
        raise SchemaValidationError("INTERVAL literal must be a string")
    m = _INTERVAL_RE.match(value)
    if m is None:
        raise SchemaValidationError(
            f"unsupported INTERVAL {value!r}; supported examples: P1D, PT3H, P2DT6H"
        )
    parts = {k: float(v) if v is not None else 0.0 for k, v in m.groupdict().items()}
    if not any(parts.values()) and value.upper() != "P0D":
        raise SchemaValidationError(
            f"unsupported INTERVAL {value!r}; supported examples: P1D, PT3H, P2DT6H"
        )
    seconds = (
        parts["weeks"] * 7 * 86_400
        + parts["days"] * 86_400
        + parts["hours"] * 3_600
        + parts["minutes"] * 60
        + parts["seconds"]
    )
    return int(seconds * 1_000_000)


def interval_micros_to_timedelta(value: int) -> timedelta:
    return timedelta(microseconds=int(value))


def _parse_datetime_string(value: str, default_timezone: str | None) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as e:
        raise SchemaValidationError(
            f"TIMESTAMPTZ value {value!r} is not a valid ISO 8601 timestamp"
        ) from e
    return _ensure_aware(dt, default_timezone)


def _ensure_aware(value: datetime, default_timezone: str | None) -> datetime:
    if value.tzinfo is not None and value.utcoffset() is not None:
        return value
    if default_timezone is None:
        raise SchemaValidationError(
            "TIMESTAMPTZ value must include a timezone offset or use a configured timezone"
        )
    try:
        tz = ZoneInfo(default_timezone)
    except ZoneInfoNotFoundError as e:
        raise SchemaValidationError(
            f"unknown timezone {default_timezone!r}"
        ) from e
    return value.replace(tzinfo=tz)
