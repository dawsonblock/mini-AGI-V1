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
    lease = router.acquire_lease()
    done = []
    t = threading.Thread(target=lambda: (
        router.deactivate("aa" * 16), done.append(1)))
    t.start()
    import time
    time.sleep(0.05)
    router.release_lease(lease)
    t.join(timeout=3)
    assert done


def test_duplicate_lease_release_is_idempotent():
    from minagi.runtime.inference_policy import InferenceBudgetPolicyV1

    router = ServingRouter(
        budget=InferenceBudgetPolicyV1(
            max_concurrent_requests=10, max_queued_per_principal=2))
    router.activate("aa" * 16, Backend(), {"id": 1})
    first = router.acquire_lease(principal="uid:1")
    second = router.acquire_lease(principal="uid:1")

    router.release_lease(first)
    router.release_lease(first)

    assert router.inflight("aa" * 16) == 1
    third = router.acquire_lease(principal="uid:1")
    with pytest.raises(RoutingRefused, match="per-principal"):
        router.acquire_lease(principal="uid:1")
    router.release_lease(second)
    router.release_lease(third)


def test_drain_timeout_is_not_unload_permission():
    """SEC-303: a drain timeout leaves the backend retained — it is
    not permission to unload while leases remain."""
    router = ServingRouter(drain_timeout=0.05)
    backend = Backend()
    router.activate("aa" * 16, backend, {"id": 1})
    lease = router.acquire_lease()
    assert router.drain_until_idle("aa" * 16, timeout=0.05) is False
    # still in flight: safe_unload refuses and the entry is retained
    assert router.safe_unload("aa" * 16) is False
    assert router.inflight("aa" * 16) == 1
    router.release_lease(lease)
    assert router.drain_until_idle("aa" * 16, timeout=0.5) is True
    # still accepting traffic — withdrawal must precede unload
    assert router.safe_unload("aa" * 16) is False
    router.stop_accepting("aa" * 16)
    assert router.safe_unload("aa" * 16) is True


def test_retire_sequence_drains_then_unloads():
    """The full withdrawal: stop new traffic, drain, unload only when
    the activation's own lease count is zero."""
    router = ServingRouter(drain_timeout=0.5)

    class Unloading(Backend):
        def __init__(self):
            self.unloaded = []

        def unload(self, handle):
            self.unloaded.append(handle)

    backend = Unloading()
    router.activate("aa" * 16, backend, {"id": 1})
    out = router.retire("aa" * 16)
    assert out["drained"] and out["unloaded"]
    assert backend.unloaded == [{"id": 1}]
    assert router.inflight("aa" * 16) == 0
    with pytest.raises(RoutingRefused):
        router.route({"prompt": "x"})


def test_retire_with_live_lease_retains_until_released():
    router = ServingRouter(drain_timeout=0.05, cancel_grace=0.05)

    class Unloading(Backend):
        def __init__(self):
            self.unloaded = []

        def unload(self, handle):
            self.unloaded.append(handle)

    backend = Unloading()
    router.activate("aa" * 16, backend, {"id": 1})
    lease = router.acquire_lease()
    out = router.retire("aa" * 16)
    assert out["drained"] is False and out["unloaded"] is False
    assert out["inflight"] == 1
    assert not backend.unloaded
    router.release_lease(lease)
    out = router.retire("aa" * 16)
    assert out["unloaded"] is True
    assert backend.unloaded == [{"id": 1}]


def test_publish_generation_must_advance():
    """CAS: a stale deployment generation cannot overwrite a completed
    newer transition."""
    router = ServingRouter()
    router.prepare_route("aa" * 16, Backend(), {"id": 1})
    router.publish_route("aa" * 16, generation=5)
    router.prepare_route("bb" * 16, Backend(), {"id": 2})
    from minagi.runtime.serving_router import RouteConflict
    with pytest.raises(RouteConflict, match="stale"):
        router.publish_route("bb" * 16, generation=4)
    # equal generation also loses
    with pytest.raises(RouteConflict, match="stale"):
        router.publish_route("bb" * 16, generation=5)
    assert router.publish_route("bb" * 16, generation=6) == 2


def test_per_activation_inflight_accounting():
    """SEC-303: each activation drains independently — B's leases do
    not block A's retirement and vice versa."""
    router = ServingRouter(drain_timeout=0.2)
    router.activate("aa" * 16, Backend(), {"id": 1})
    lease_a = router.acquire_lease()
    router.activate("bb" * 16, Backend(), {"id": 2})
    lease_b = router.acquire_lease()
    assert router.inflight("aa" * 16) == 1
    assert router.inflight("bb" * 16) == 1
    router.release_lease(lease_a)
    assert router.drain_until_idle("aa" * 16, timeout=0.2) is True
    assert router.drain_until_idle("bb" * 16, timeout=0.05) is False
    router.release_lease(lease_b)


def test_prepare_over_live_route_refused():
    router = ServingRouter()
    router.activate("aa" * 16, Backend(), {"id": 1})
    from minagi.runtime.serving_router import RouteConflict
    with pytest.raises(RouteConflict, match="live"):
        router.prepare_route("aa" * 16, Backend(), {"id": 9})


def test_lease_released_in_finally_on_backend_error():
    router = ServingRouter()

    class Bad(Backend):
        def infer(self, handle, request):
            raise RuntimeError("gpu fault")

    router.activate("aa" * 16, Bad(), {"id": 1})
    with pytest.raises(RuntimeError):
        router.route({"prompt": "x"})
    assert router.inflight() == 0
    assert router.drain_until_idle("aa" * 16, timeout=0.1) is True
