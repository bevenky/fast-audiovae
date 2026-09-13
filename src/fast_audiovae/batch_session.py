"""Serialize stateless batch calls that reuse native matrix packing scratch."""
from __future__ import annotations

import threading


class LockedBatchSession:
    """Keep synchronous ORT session execution exclusive without retaining history.

    Only selected native batch sessions need this proxy. Metadata access passes
    through to the session; asynchronous execution is deliberately unsupported.
    """

    def __init__(self, session):
        self._session = session
        self._lock = threading.Lock()

    def _run(self, method, *args, **kwargs):
        with self._lock:
            return getattr(self._session, method)(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self._run("run", *args, **kwargs)

    def run_with_iobinding(self, *args, **kwargs):
        return self._run("run_with_iobinding", *args, **kwargs)

    def run_with_ort_values(self, *args, **kwargs):
        return self._run("run_with_ort_values", *args, **kwargs)

    def run_with_ortvaluevector(self, *args, **kwargs):
        return self._run("run_with_ortvaluevector", *args, **kwargs)

    def run_async(self, *args, **kwargs):
        raise NotImplementedError("Selected batch sessions support synchronous execution only")

    def __getattr__(self, name):
        value = getattr(self._session, name)
        if name.startswith("run") and callable(value):
            raise NotImplementedError("Unsupported batch execution method: " + name)
        return value
