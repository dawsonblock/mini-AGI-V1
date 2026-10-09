from kvcontinual.continual.backends import OpenAIBackend
from kvcontinual.continual.integrations.openai_compatible import OpenAICompatibleClient


def test_llamacpp_capabilities():
    b = OpenAIBackend(OpenAICompatibleClient("http://127.0.0.1:1/v1"), backend_kind="llama.cpp")
    c = b.capabilities()
    assert c.metal_acceleration
    assert c.lora_load
    assert c.lora_hot_switch
    assert not c.recurrent_state
