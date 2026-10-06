import json
from minagi.shadow_live import ShadowLearningBuffer
from minagi.tokenizer import ByteTokenizer


def test_shadow_learning_is_append_only_and_persistent(tmp_path):
    b=ShadowLearningBuffer(tmp_path)
    r=b.feed('<user>hello</user>',ByteTokenizer())
    assert len(r)==1 and b.records==1 and b.tokens>0
    line=json.loads((tmp_path/'pending.jsonl').read_text().strip())
    assert line['status']=='pending' and len(line['sha256'])==64
    b2=ShadowLearningBuffer(tmp_path)
    assert b2.records==1 and b2.tokens==b.tokens
    st=b2.state()
    assert st['mode']=='shadow' and st['pending']==b2.tokens
    assert st['chunk']==0 and st['steps']==0
