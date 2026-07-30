"""Shared schema base classes.

The team convention is **camelCase on the wire, snake_case in Python**.
`CamelModel` centralizes that rule so individual schemas stay clean and every
response is serialized consistently. See `docs/API_SPEC.md`.
"""

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, PlainSerializer
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """Base model that (de)serializes camelCase JSON but uses snake_case fields.

    - Outgoing JSON uses camelCase aliases (``by_alias`` is the default here).
    - Incoming JSON is accepted in either camelCase or snake_case.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )


def _to_utc_iso_z(value: datetime) -> str:
    """Render a datetime as ``2026-07-24T11:22:57Z``.

    Pydantic's default output is ``2026-07-24T11:22:57.123456+00:00``, which does
    not match the timestamp format documented in `docs/API_SPEC.md`.
    """
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ISO-8601 UTC timestamp, serialized with a `Z` suffix and second precision.
UtcTimestamp = Annotated[datetime, PlainSerializer(_to_utc_iso_z, return_type=str)]
