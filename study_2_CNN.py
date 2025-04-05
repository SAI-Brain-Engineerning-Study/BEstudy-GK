import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 데이터 로드 및 전처리
# 이미지를 PyTorch 텐서로 변환 (0255 → 01 사이 값으로 변경)
# 데이터 정규화 (평균 0, 표준편차 0.5로 맞춤)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
# train=True >  학습데이터
# train=False > 테스트 데이터

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # 입력 이미지(1채널) > 3×3 필터 32개 적용
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 이전 층의 출력(32채널) → 3×3 필터 64개 적용
        self.pool = nn.MaxPool2d(2, 2) # 2×2 최대 풀링(이미지 크기를 절반으로 줄임)
        self.dropout1 = nn.Dropout(0.25) # 25% 확률로 뉴런 제거 (과적합 방지)
        self.fc1 = nn.Linear(64 * 14 * 14, 128) # 출력 128개
        self.dropout2 = nn.Dropout(0.5) # 50% 확률로 뉴런 제거
        self.fc2 = nn.Linear(128, 10) # 숫자 0~9 분류

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
    # 모델 및 손실 함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 가능하면 CUDA, 아니면 CPU
model = CNN().to(device)
criterion = nn.CrossEntropyLoss() # 손실 함수로 교차 엔트로피(CrossEntropyLoss) 사용
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam 옵티마이저 사용 (학습률 0.001)

# 학습
# optimizer.zero_grad() → 이전 미분값 초기화
# outputs = model(images) → 모델에 이미지 입력하여 예측값 출력
# loss = criterion(outputs, labels) → 손실 계산
# loss.backward() → 손실을 역전파
# optimizer.step() → 가중치 업데이트

num_epochs = 30
early_stop_count = 0
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
 # 검증
 # model.eval() → 모델을 평가 모드로 설정
 # torch.no_grad() → 그래디언트 계산 비활성화 (속도 향상)
 # _, predicted = torch.max(outputs, 1) → 가장 높은 확률의 클래스를 예측
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(test_loader)
    accuracy = correct / total
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    # Early Stopping - 성능이 10번 연속 개선되지 않으면 학습 종료
    if val_loss < best_loss:
        best_loss = val_loss
        early_stop_count = 0
        torch.save(model.state_dict(), "best_mnist_model.pth")
    else:
        early_stop_count += 1

    if early_stop_count >= 10:
        print("Early stopping triggered")
        break

    # 테스트 성능 평가 - 최고 성능 모델로 최종 정확도 출력
model.load_state_dict(torch.load("best_mnist_model.pth"))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Final Test Accuracy: {correct / total:.4f}')