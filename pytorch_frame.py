import torch

data = [
    [1, 2],
    [3, 4]
]

x = torch.tensor(data)
print(x.is_cuda)

#GPU로 옮김
x = x.cuda()
print(x.is_cuda)

#CPU로 옮김
x = x.cpu()
print(x.is_cuda)