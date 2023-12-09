import torch
a = torch.tensor([[1, 2]])
b = torch.tensor([[2, 1]])
c = torch.cat([a, b], 0)
print(c)