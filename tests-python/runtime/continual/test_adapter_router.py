import json
from kvcontinual.continual.adapter_router import AdapterDescriptor, AdapterRouter


def test_router_prefers_relevant_adapter():
    r = AdapterRouter([
        AdapterDescriptor("code", "/code", "python coding debugging", ("code", "python"), backend_id=0),
        AdapterDescriptor("medical", "/medical", "medical terminology", ("health",), backend_id=1),
    ])
    out = r.route("debug this python function", top_k=1)
    assert out and out[0][0].adapter_id == "code"


def test_router_loads_json(tmp_path):
    p = tmp_path / "adapters.json"
    p.write_text(json.dumps([{"adapter_id":"code","path":"/a","description":"python code","backend_id":3}]))
    r = AdapterRouter.from_json(p)
    assert r.adapters[0].backend_id == 3
