import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils

device = torch.device('cuda')

# 데이터셋 로딩 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1] 범위
])
data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(data, batch_size=64, shuffle=True)

# 생성자 정의
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Tanh()
        )
    def forward(self, z):
        out = self.gen(z)
        return out.view(-1, 1, 28, 28)

# 판별자 정의
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dis = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.dis(x)

# 모델, 손실함수, 옵티마이저 초기화
G = Generator().to(device)
D = Discriminator().to(device)
loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)

# 이미지 시각화 함수
def show_generated_images(images, epoch, num_images=25):
    images = images[:num_images].detach().cpu()
    grid = vutils.make_grid(images, nrow=5, normalize=True, pad_value=1)
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.title(f"Generated Images (Epoch {epoch+1})")
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.show()

# 학습 루프
num_epochs = 200
for epoch in range(num_epochs):
    for batch_idx, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)

        # 1. 노이즈 z 생성 및 가짜 이미지 생성
        z = torch.randn(imgs.size(0), 100, device=device)
        fake_imgs = G(z)

        # 2. 진짜/가짜 라벨 생성
        real_labels = torch.ones(imgs.size(0), 1, device=device)
        fake_labels = torch.zeros(imgs.size(0), 1, device=device)

        # 3. 판별자 학습
        real_loss = loss_fn(D(imgs), real_labels)
        fake_loss = loss_fn(D(fake_imgs.detach()), fake_labels)
        D_loss = real_loss + fake_loss

        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        # 4. 생성자 학습
        G_loss = loss_fn(D(fake_imgs), real_labels)  # 생성자는 진짜처럼 보이길 원함

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()
    # 5. epoch 종료 후 결과 처음과 끝만 출력
    if epoch == 0 or epoch == 199: 
        print(f"Epoch [{epoch+1}/{num_epochs}]  D_loss: {D_loss.item():.4f}  G_loss: {G_loss.item():.4f}")
        show_generated_images(fake_imgs, epoch)
