"""HTTP layer: routers and request/response wiring only.

Routers stay thin — validate input, call a service, shape the response. Business
logic lives in `app/services` and `app/domain`.
"""
