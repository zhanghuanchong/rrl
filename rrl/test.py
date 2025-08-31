import torch

print('mps available: ' + str(torch.backends.mps.is_available()))
print('mps builtin: ' + str(torch.backends.mps.is_built()))

print('cuda builtin: ' + str(torch.backends.cuda.is_built()))
