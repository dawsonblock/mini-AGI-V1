from egai.cognition.model import FrozenModelIdentity
from egai.cognition.backends import OpenAICompatibleFrozenModel

# Replace these with the real digests of your local immutable model artifacts.
identity=FrozenModelIdentity(
    model_name='your-local-qwen',
    files=(('model.safetensors','sha256:'+'0'*64),('tokenizer.json','sha256:'+'1'*64)),
    config_digest='sha256:'+'2'*64,
    tokenizer_digest='sha256:'+'3'*64,
    runtime_family='openai-compatible',
)
model=OpenAICompatibleFrozenModel('http://127.0.0.1:8000','your-local-qwen',identity)
print('model digest:',model.model_digest)
# print(model.generate('Return the number 4.'))
