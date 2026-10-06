import json
from dataclasses import asdict
from .model import SkillManifest
from .memory import LearnedProcedure

def procedure_to_manifest(p:LearnedProcedure,version='1'):
    return SkillManifest(
        skill_id=p.procedure_id,version=version,purpose=f'Learned procedure for {p.task_kind}',
        activation_conditions=(p.trigger_text,),preconditions=(),contraindications=(),inputs=('task',),outputs=('answer',),
        implementation={'handler':'prompt_procedure','procedure_text':p.procedure_text},permissions=(),
        resource_limits={'max_invocations_per_task':1},termination_conditions=('answer produced',),
        success_postconditions=('external verifier accepts result',),verifier={'handler':'external'},
        known_failure_modes=(),supporting_evidence=p.supporting_episode_ids)

def bundle_manifests(procedures):
    return tuple(procedure_to_manifest(p) for p in procedures)
