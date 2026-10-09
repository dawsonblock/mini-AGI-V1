import pytest
from egai.cognition.model import FrozenModel,FrozenModelIdentity
from egai.cognition.agent import SandboxAdaptiveAgent
from egai.bench.sequential import SequentialExperiment
from egai.bench.engine import TaskCase
from egai.cognition.backends import LlamaCppCLIModel


class ChangingModel(FrozenModel):
    def __init__(self):self.ident=FrozenModelIdentity('frozen',());self.count=0
    @property
    def model_digest(self):return self.ident.digest
    def generate(self,prompt):
        self.count+=1
        if self.count==2:self.ident=FrozenModelIdentity('changed',())
        return 'a'


def test_change_during_final_evaluation_is_caught():
    model=ChangingModel();agent=SandboxAdaptiveAgent(model)
    experiment=SequentialExperiment([], [TaskCase('future','future','a','a')],lambda a,b:float(a==b),checkpoints=(0,))
    with pytest.raises(RuntimeError,match='frozen model identity changed'):experiment.run(model,agent)


def test_local_checkpoint_mutation_changes_model_identity(tmp_path):
    p=tmp_path/'weights.gguf';p.write_bytes(b'original')
    model=LlamaCppCLIModel('llama-cli',p);original=model.model_digest
    p.write_bytes(b'changed');assert model.model_digest!=original
