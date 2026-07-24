"""Shared schema base classes.

The team convention is **camelCase on the wire, snake_case in Python**.
`CamelModel` centralizes that rule so individual schemas stay clean and every
response is serialized consistently. See `docs/API_SPEC.md`.
"""

from pydantic import BaseModel, ConfigDict
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
