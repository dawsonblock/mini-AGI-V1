from dataclasses import dataclass
import re
TOKEN_RE=re.compile(r"[a-zA-Z0-9_+-]+")
def _tok(x): return set(TOKEN_RE.findall(str(x).lower()))
def _norm(x): return ' '.join(TOKEN_RE.findall(str(x).lower()))
def _shingles(x,n=5):
    s=_norm(x).replace(' ','_')
    return {s[i:i+n] for i in range(max(0,len(s)-n+1))} or ({s} if s else set())
def _jac(a,b): return len(a&b)/max(len(a|b),1)
@dataclass(frozen=True)
class LeakageReport:
    exact_input_overlap:tuple[str,...]
    exact_case_overlap:tuple[str,...]
    id_overlap:tuple[str,...]
    near_input_pairs:tuple[tuple[str,str,float],...]=()
    normalized_input_pairs:tuple[tuple[str,str],...]=()
    @property
    def clean(self):return not (self.exact_input_overlap or self.exact_case_overlap or self.id_overlap or self.near_input_pairs or self.normalized_input_pairs)
def detect_leakage(experience_cases,evaluation_cases,near_threshold=.95):
    exp_ids={c.case_id for c in experience_cases}; ev_ids={c.case_id for c in evaluation_cases}
    exp_in={c.input_digest for c in experience_cases}; ev_in={c.input_digest for c in evaluation_cases}
    exp_case={c.case_digest for c in experience_cases};ev_case={c.case_digest for c in evaluation_cases}
    near=[]; normalized=[]
    for a in experience_cases:
        for b in evaluation_cases:
            if a.input_digest==b.input_digest: continue
            if _norm(a.input)==_norm(b.input): normalized.append((a.case_id,b.case_id)); continue
            if near_threshold is not None:
                token_sim=_jac(_tok(a.input),_tok(b.input)); char_sim=_jac(_shingles(a.input),_shingles(b.input)); sim=max(token_sim,char_sim)
                if sim>=near_threshold: near.append((a.case_id,b.case_id,round(sim,6)))
    return LeakageReport(tuple(sorted(exp_in&ev_in)),tuple(sorted(exp_case&ev_case)),tuple(sorted(exp_ids&ev_ids)),tuple(sorted(near)),tuple(sorted(normalized)))
