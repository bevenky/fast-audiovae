"""Selected batch sessions serialize every supported synchronous entry point."""
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations_with_replacement
import threading
import time

import pytest

from fast_audiovae.batch_session import LockedBatchSession


METHODS = ("run", "run_with_iobinding", "run_with_ort_values", "run_with_ortvaluevector")


class Session:
    def __init__(self):
        self.calls = []
        self.result = object()
        self.metadata = object()
        self.run_label = "metadata property"
        self.error = None

    def get_inputs(self):
        return self.metadata

    def _execute(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        if self.error is not None:
            raise self.error
        return self.result

    def run(self, *args, **kwargs):
        return self._execute("run", *args, **kwargs)

    def run_with_iobinding(self, *args, **kwargs):
        return self._execute("run_with_iobinding", *args, **kwargs)

    def run_with_ort_values(self, *args, **kwargs):
        return self._execute("run_with_ort_values", *args, **kwargs)

    def run_with_ortvaluevector(self, *args, **kwargs):
        return self._execute("run_with_ortvaluevector", *args, **kwargs)

    def run_async(self, *args, **kwargs):
        raise AssertionError("Asynchronous work must never reach the session")

    def run_future_api(self, *args, **kwargs):
        raise AssertionError("Unknown execution must never reach the session")


@pytest.mark.parametrize("method", METHODS)
def test_execution_forwards_arguments_and_return_value(method):
    session = Session()
    proxy = LockedBatchSession(session)
    argument, option = object(), object()
    assert getattr(proxy, method)(None, argument, run_options=option) is session.result
    assert session.calls == [(method, (None, argument), {"run_options": option})]


@pytest.mark.parametrize("first,second", combinations_with_replacement(METHODS, 2))
def test_real_callers_share_one_lock_across_execution_methods(first, second):
    class ScratchSession(Session):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.maximum_active = 0
            self.observation_lock = threading.Lock()

        def _execute(self, method, token):
            with self.observation_lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            try:
                # Release the GIL, as a native call using shared scratch does.
                time.sleep(.01)
                return token
            finally:
                with self.observation_lock:
                    self.active -= 1

    session = ScratchSession()
    proxy = LockedBatchSession(session)
    start = threading.Barrier(2)

    def call(method, token):
        start.wait(timeout=2)
        return getattr(proxy, method)(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(call, first, "a")
        b = pool.submit(call, second, "b")
        assert a.result(timeout=2) == "a"
        assert b.result(timeout=2) == "b"
    assert session.maximum_active == 1
    assert session.active == 0


def test_error_propagates_and_lock_is_released_for_another_caller():
    session = Session()
    proxy = LockedBatchSession(session)
    session.error = ValueError("inference failed")
    with pytest.raises(ValueError) as caught:
        proxy.run(None, {})
    assert caught.value is session.error
    session.error = None
    results = []
    worker = threading.Thread(target=lambda: results.append(proxy.run_with_iobinding("binding")), daemon=True)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive(), "Failed execution left the shared lock held"
    assert results == [session.result]


def test_metadata_delegates_without_exposing_unknown_execution():
    session = Session()
    proxy = LockedBatchSession(session)
    assert proxy.get_inputs() is session.metadata
    assert proxy.metadata is session.metadata
    assert proxy.run_label == "metadata property"
    with pytest.raises(AttributeError):
        _ = proxy.missing_property
    with pytest.raises(NotImplementedError, match="synchronous"):
        proxy.run_async(None, {}, callback=object())
    with pytest.raises(NotImplementedError, match="run_future_api"):
        proxy.run_future_api()
    assert session.calls == []
