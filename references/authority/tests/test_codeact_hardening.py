import pytest
from minagi.codeact import CapabilitySandbox, ActionRejected


def test_callable_cannot_enter_through_variables():
    vm = CapabilitySandbox()
    with pytest.raises(ActionRejected):
        vm.run("x()", {"x": lambda: 1})


def test_resource_amplification_is_rejected_before_result_allocation():
    vm = CapabilitySandbox(max_collection=100)
    with pytest.raises(ActionRejected):
        vm.run("x = 'a' * 1000000")
    with pytest.raises(ActionRejected):
        vm.run("x = 2 ** 1000", variables={})


def test_tool_budget_and_plain_output_boundary():
    vm = CapabilitySandbox({"inc": lambda x: x + 1}, max_tool_calls=2)
    r = vm.run("a=inc(1)\nb=inc(a)\n_result=b")
    assert r.result == 3 and len(r.tool_calls) == 2
    with pytest.raises(ActionRejected):
        vm.run("a=inc(1)\nb=inc(2)\nc=inc(3)")
    bad = CapabilitySandbox({"bad": lambda: object()})
    with pytest.raises(ActionRejected):
        bad.run("x=bad()")


def test_bool_short_circuit_does_not_call_tool():
    calls=[]
    vm=CapabilitySandbox({"touch": lambda: calls.append(1) or True})
    vm.run("x = False and touch()\ny = True or touch()")
    assert calls == []
