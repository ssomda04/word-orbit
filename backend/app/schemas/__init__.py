"""Pydantic request/response schemas (the API's wire contract).

All schemas serialize to **camelCase** JSON (see `docs/API_SPEC.md`) while the
Python attributes stay snake_case. This is handled once in `CamelModel`.
"""
