"""Process-safe atomic file writes (write temp -> ``os.replace`` onto the destination).

``atomic_replace`` is the shared primitive: a hoist of ``_atomic_replace`` from
``scripts/experiment_014_run.py`` (#014, commit ``27206c15d9``, the round-3-corrected form)
into shared code (#2336). Its body is verbatim from the donor except for exactly one
mechanical change: the cleanup warning logs via ``(logger or _LOGGER)`` so a caller with
its own logger contract (e.g. ``experiment_014.run``) can keep it.

Three load-bearing properties (do not weaken any of them):

1. The temp name is PROCESS-UNIQUE: ``f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"``.
   Concurrent workers writing ONE shared destination must not share a temp name, or one
   worker's ``os.replace`` consumes the shared temp and every later worker dies
   ``FileNotFoundError`` (the #014 grid crash, 2026-08-16).
2. The temp lives in the DESTINATION's own directory — a cross-filesystem ``os.replace``
   is not atomic (never route through ``/tmp``).
3. On ``BaseException`` the temp is best-effort unlinked (``try tmp.unlink(missing_ok=True)
   / except OSError -> log``) followed by a bare ``raise``, so a cleanup failure never
   displaces the ORIGINAL exception.

Module-level imports are stdlib-only so lightweight scripts can import this without
dragging the ML stack. ``numpy`` is imported lazily inside ``savez_atomic``.

Two names ship: ``atomic_replace``, the primitive, and ``savez_atomic``, the one
wrapper the pipeline steps call. The donor module's other typed wrappers
(json / jsonl / text / bytes / torch / npy) are not carried here, since nothing in
this release called them, and ``atomic_replace`` is all a new one needs.
"""

from __future__ import annotations

import contextlib
import logging
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)  # "context_answer.atomic_io"


@contextlib.contextmanager
def atomic_replace(path: Path, *, logger: logging.Logger | None = None) -> Iterator[Path]:
    """Yield a PROCESS-UNIQUE same-directory temp path; ``os.replace`` it
    onto ``path`` on success, best-effort unlink it on failure (a cleanup
    failure never masks the original exception).

    The temp name embeds pid + a uuid fragment: concurrent workers writing
    identical content to ONE shared destination must not share a temp name,
    or one worker's replace consumes the shared temp and every later worker
    dies ``FileNotFoundError`` (#014 grid crash 2026-08-16 05:36Z, rc=1).
    Same-dir keeps the replace atomic (one filesystem — never route through
    /tmp); unlink-on-failure keeps orphan ``*.tmp`` residue out of the
    out-root (the upload-verifier residue-sweep surface). Concurrent
    same-content writes stay safe/idempotent: last atomic replace wins with
    identical bytes.

    ``logger`` routes the suppressed-cleanup warning to the caller's logger
    (default: this module's).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        # Best-effort cleanup: a failed unlink (PermissionError / non-ENOENT
        # OSError) must never displace the ORIGINAL write/replace exception —
        # the bare ``raise`` re-raises it with its traceback intact. Only the
        # SECONDARY cleanup error is suppressed (logged); the fault itself
        # stays loud (#014 r3 finding 1, ``cleanup-can-mask-original``).
        try:
            tmp.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            (logger or _LOGGER).warning(
                "cleanup unlink of %s failed (%s); propagating original exception",
                tmp,
                cleanup_exc,
            )
        raise


def savez_atomic(
    path: Path | str,
    *,
    compressed: bool = False,
    logger: logging.Logger | None = None,
    **arrays: Any,
) -> None:
    """Atomically ``np.savez``/``np.savez_compressed`` ``arrays`` to ``path``.

    Writes through an OPEN FILE HANDLE, never the temp path: ``np.savez`` APPENDS
    ``.npz`` to a path-name lacking it, so a path-form ``np.savez(tmp)`` would write
    ``<tmp>.npz`` and the subsequent ``os.replace`` would die ``FileNotFoundError``
    (the yielded temp always ends ``.tmp``). numpy never appends to handles.

    NOTE: an array keyed ``compressed`` or ``logger`` collides with the keyword-only
    parameters (``np.savez`` has the analogous trap with its own kwargs) — pass such
    arrays by building the dict first and renaming the key, or write through
    ``atomic_replace`` directly.
    """
    import numpy as np  # lazy: keep module-level imports stdlib-only

    path = Path(path)
    with atomic_replace(path, logger=logger) as tmp, tmp.open("wb") as fh:
        if compressed:
            np.savez_compressed(fh, **arrays)
        else:
            np.savez(fh, **arrays)
