from dataclasses import dataclass
import re

TOKEN_RE=re.compile(r"[a-zA-Z0-9_+-]+")
def _tok(x): return set(TOKEN_RE.findall(str(x).lower()))

@dataclass(frozen=True)
class LeakageReport:
    exact_input_overlap:tuple[str,...]
    exact_case_overlap:tuple[str,...]
    id_overlap:tuple[str,...]
    near_input_pairs:tuple[tuple[str,str,float],...]=()
    @property
    def clean(self):return not (self.exact_input_overlap or self.exact_case_overlap or self.id_overlap or self.near_input_pairs)

def detect_leakage(experience_cases,evaluation_cases,near_threshold=.95):
    exp_ids={c.case_id for c in experience_cases}; ev_ids={c.case_id for c in evaluation_cases}
    exp_in={c.input_digest for c in experience_cases}; ev_in={c.input_digest for c in evaluation_cases}
    exp_case={c.case_digest for c in experience_cases};ev_case={c.case_digest for c in evaluation_cases}
    near=[]
    if near_threshold is not None:
        for a in experience_cases:
            ta=_tok(a.input)
            for b in evaluation_cases:
                tb=_tok(b.input)
                if not ta and not tb: sim=1.0
                else: sim=len(ta&tb)/max(len(ta|tb),1)
                if sim>=near_threshold and a.input_digest!=b.input_digest: near.append((a.case_id,b.case_id,round(sim,6)))
    return LeakageReport(tuple(sorted(exp_in&ev_in)),tuple(sorted(exp_case&ev_case)),tuple(sorted(exp_ids&ev_ids)),tuple(sorted(near)))
