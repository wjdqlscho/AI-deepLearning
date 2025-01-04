import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import time
import matplotlib.pyplot as plt

batch_size = 128
learning_rate = 0.001

#딥러닝 모델 정의
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

optimizer_names = ["SGD", "Adagrad", "RMSprop", "Adadelta", "Adam"] #최적화 알고리즘 대표 5개 선정
models = [Model().cuda() for i in range(5)]
optimizers = [
    optim.SGD(models[0].parameters(), lr=learning_rate),
    optim.Adagrad(models[1].parameters(), lr=learning_rate),
    optim.RMSprop(models[2].parameters(), lr=learning_rate),
    optim.Adadelta(models[3].parameters(), lr=learning_rate),
    optim.Adam(models[4].parameters(), lr=learning_rate)
]

# 로깅(logging)
train_losses = [[] for _ in range(5)]
test_losses = [[] for _ in range(5)]
train_accuracies = [[] for _ in range(5)]
test_accuracies = [[] for _ in range(5)]

#학습
criterion = nn.CrossEntropyLoss(reduction="mean")
n_epoch = 20
log_step = 100

# 각 실험(experiment)에 대하여
for exp in range(5):
    # 현재의 실험 설정에 대하여 출력
    print("=================================================")
    print(f"[Experiment {exp + 1}]")
    print(f"Optimizer: {optimizer_names[exp]}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Total number of epochs: {n_epoch}")
    start_time = time.time()

    # 현재 실험에서 사용할 모델(model)과 최적화 방법(optimizer)을 선택
    model = models[exp]
    optimizer = optimizers[exp]

    # 반복(epoch)하여 학습 수행
    for epoch in range(n_epoch):
        print(f"[Epoch: {epoch + 1}]")

        # 학습(training)
        model.train()
        total = 0
        running_loss = 0.0
        running_corrects = 0.0
        for i, batch in enumerate(train_dataloader):
            # 현재 배치의 이미지와 레이블 꺼내기
            inputs, targets = batch
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, targets)

            loss.backward()  # 기울기 계산
            optimizer.step()  # 계산된 기울기를 이용해 가중치 업데이트

            total += targets.shape[0]
            running_loss += (loss.item() * targets.shape[0])
            running_corrects += torch.sum(preds == targets.data)

        train_loss = running_loss / total
        train_accuracy = running_corrects / total
        print(f'Train loss: {train_loss:.6f}, train accuracy: {train_accuracy * 100.:.2f}%')

        # 테스트(test)
        model.eval()
        total = 0
        running_loss = 0.0
        running_corrects = 0.0
        for i, batch in enumerate(test_dataloader):
            # 현재 배치의 이미지와 레이블 꺼내기
            inputs, targets = batch
            inputs, targets = inputs.cuda(), targets.cuda()

            # 학습 없이 정확도를 평가하므로 기울기(gradient) 추적 제외
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, targets)

            total += targets.shape[0]
            running_loss += (loss.item() * targets.shape[0])
            running_corrects += torch.sum(preds == targets.data)

        test_loss = running_loss / total
        test_accuracy = running_corrects / total
        print(f'Test loss: {test_loss:.6f}, test accuracy: {test_accuracy * 100.:.2f}%')

        # 로깅(logging)
        train_losses[exp].append(train_loss)
        test_losses[exp].append(test_loss)
        train_accuracies[exp].append(train_accuracy)
        test_accuracies[exp].append(test_accuracy)
        print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")

markers = [".", "v", "^", "s", "*"]
plt.figure(figsize=(20, 10))
epochs = [[i] for i in range(1, 21)]

plt.subplot(1, 2, 1)
for exp in range(5):
    train_accuracy = [x.cpu() for x in train_accuracies[exp]]
    plt.plot(epochs, train_accuracy, marker=markers[exp], label=optimizer_names[exp])
plt.xlabel("Epoch")
plt.ylabel("Train Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
for exp in range(5):
    plt.plot(epochs, train_losses[exp], marker=markers[exp], label=optimizer_names[exp])
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.legend()

plt.show()