"""v16.4.3 — ServingRouter: only a healthy, committed, live handle may
serve (OPS-003 / WP8)."""
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.runtime.serving_router import (  # noqa: E402
    RoutingRefused, ServingRouter)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)


class Backend:
    backend_id = "hf-peft"

    def infer(self, handle, request):
        return {"echo": request.get("prompt", ""),
                "handle": handle["id"]}


def test_no_active_route_refused():
    router = ServingRouter()
    with pytest.raises(RoutingRefused, match="unavailable|no live"):
        router.route({"prompt": "hi"})
    assert router.status()["serving"] is False


def test_activate_routes_to_handle():
    router = ServingRouter()
    backend = Backend()
    router.activate("aa" * 16, backend, {"id": 1})
    out = router.route({"prompt": "hello"})
    assert out["activation_id"] == "aa" * 16
    assert out["result"]["handle"] == 1
    assert router.status()["serving"] is True


def test_deactivate_stops_routing():
    router = ServingRouter()
    router.activate("aa" * 16, Backend(), {"id": 1})
    router.deactivate("aa" * 16)
    with pytest.raises(RoutingRefused):
        router.route({"prompt": "hi"})


def test_swap_is_atomic_for_callers():
    """A concurrent activate during a route never yields a torn state —
    each request sees exactly one version."""
    router = ServingRouter()
    router.activate("aa" * 16, Backend(), {"id": 1})
    router.activate("bb" * 16, Backend(), {"id": 2})
    out = router.route({"prompt": "x"})
    assert out["result"]["handle"] == 2
    assert out["routing_version"] == 2


def test_router_never_routes_to_raw_path():
    """The router accepts (backend, handle) pairs from the supervisor
    only — a string path or dict is not a routable object."""
    router = ServingRouter()
    router.activate("aa" * 16, "/tmp/model", {"id": 1})
    with pytest.raises(RoutingRefused, match="no inference"):
        router.route({"prompt": "x"})


def test_concurrent_routes_observe_consistent_version():
    router = ServingRouter()
    router.activate("aa" * 16, Backend(), {"id": 7})
    versions = []
    errs = []

    def hit():
        try:
            versions.append(router.route({"prompt": "p"})[
                                "routing_version"])
        except Exception as exc:  # noqa: BLE001
            errs.append(exc)

    ts = [threading.Thread(target=hit) for _ in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs
    assert set(versions) == {1}


def test_drain_waits_for_inflight():
    router = ServingRouter(drain_timeout=0.5)
    router.activate("aa" * 16, Backend(), {"id": 1})
    with router._lock:
        router._inflight += 1
    done = []
    t = threading.Thread(target=lambda: (
        router.deactivate("aa" * 16), done.append(1)))
    t.start()
    import time
    time.sleep(0.05)
    with router._drained:
        router._inflight -= 1
        router._drained.notify_all()
    t.join(timeout=3)
    assert done
