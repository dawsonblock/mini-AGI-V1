from egai.cognition.backends import MLXLMFrozenModel
model=MLXLMFrozenModel('/path/to/local/model',model_name='local-frozen-model')
print('model digest:',model.model_digest)
# print(model.generate('Explain why 2+2=4.'))
