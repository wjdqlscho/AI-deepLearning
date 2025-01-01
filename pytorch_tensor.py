# 파이토치의 tensor는 기능적으로 numpy와 유사함
# 다차원 배열을 처리하기 위한 적합한 자료구조

import torch

tensor = torch.rand(3, 4)

print(tensor)
print(f"Shape : {tensor.shape}")
print(f"dtype : {tensor.dtype}")
print(f"Shape : {tensor.device}")