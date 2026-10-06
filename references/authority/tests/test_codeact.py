import pytest
from minagi.codeact import CapabilitySandbox, ActionRejected


def test_codeact_control_flow_and_tool_capability():
    calls=[]
    def square(x): calls.append(x); return x*x
    vm=CapabilitySandbox({'square':square})
    r=vm.run('''
xs = [1,2,3,4]
out = []
for x in xs:
    out = out + [square(x)]
_result = sum(out)
print(_result)
''')
    assert r.result == 30
    assert calls == [1,2,3,4]
    assert [c['tool'] for c in r.tool_calls] == ['square']*4
    assert r.prints == ['30']


def test_codeact_has_no_ambient_python():
    vm=CapabilitySandbox()
    with pytest.raises(ActionRejected): vm.run('import os')
    with pytest.raises(ActionRejected): vm.run('(1).__class__')
    with pytest.raises(ActionRejected): vm.run('open("x","w")')


def test_codeact_bounds_loops():
    vm=CapabilitySandbox(max_loop=5)
    with pytest.raises(ActionRejected): vm.run('for i in range(100):\n    x=i')
