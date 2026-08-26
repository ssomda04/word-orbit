"""Failures raised while reading an artifact root.

Deliberately *not* an ``AppError``: a broken or mismatched artifact root is a
configuration failure that should stop the process, not a per-request error
mapped onto the documented HTTP envelope — the same reasoning as
``FastTextConfigurationError``.

The secrecy rule for every message in this package
--------------------------------------------------
An artifact root maps **answer words** to files. The answer of a game in
progress must never appear in an exception message, because the message renders
into the traceback that ``app.main``'s ``INTERNAL_ERROR`` handler logs
(AGENTS.md). So no message here may interpolate an answer, and there is no
"answer not found" exception type at all — a missing answer is reported by
returning ``None``, which cannot carry a word.

What messages *may* name is the ``artifact_id``: the sha256 of the normalized
answer, which is also the on-disk directory name. It is not a secret and it is
the only handle that lets an operator find the offending file, so identifying a
failure by id is both safe and necessary. It is not a cryptographic protection
either — anyone holding the root also holds ``manifest.json``, which lists the
answers in plain text. Its purpose is narrower: keeping plaintext answers out of
file paths, and therefore out of anything that echoes a path.
"""


class ArtifactError(RuntimeError):
    """The artifact root cannot be read, validated, or trusted."""
