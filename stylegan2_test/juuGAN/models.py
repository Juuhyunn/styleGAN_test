from django.db import models
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
# Create your models here.


class JuuGAN:
    def __init__(self):
        self.epochs = 50
        self.batch_size = 100
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print("Using Device:", self.device)

    def process(self):
        train = datasets.FashionMNIST(
            './data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train,
            batch_size=self.batch_size,
            shuffle=True
        )
        # generator 만들기
        generator = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )
        # discriminator 만들기
        discriminator = nn.Sequential(
            nn.Linear(784, 256),
            # 양의 기울기만 전달했던 기존의 ReLU와 달리, 약간의 음의 기울기도 다음 layer로 전달한다.
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        # 모델의 가중치를 지정한 장치로 보내기 (CUDA)
        generator = generator.to(self.device)
        discriminator = discriminator.to(self.device)
        # 이진 크로스 엔트로피 (Binary cross entropy) 오차 함수와 생성자와 판별자를 최적화할 Adam 모듈
        criterion = nn.BCELoss()    # loss 함수
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)     # optimizer
        g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)         # optimizer

        total_step = len(train_loader)      # step을 정해조!
        for epoch in range(self.epochs):
            for i, (images, _) in enumerate(train_loader):
                images = images.reshape(self.batch_size, -1).to(self.device)

                # '진짜'와 '가짜' 레이블 생성
                real_labels = torch.ones(self.batch_size, 1).to(self.device)  # [1,1,1...]
                fake_labels = torch.zeros(self.batch_size, 1).to(self.device)  # [0.0,0...]

                # 판별자가 진짜 이미지를 진짜로 인식하는 오차를 예산
                outputs = discriminator(images)  # 진짜 이미지를 discriminator의 입력으로 제공
                d_loss_real = criterion(outputs, real_labels)   # loss 함수
                real_score = outputs

                # 무작위 텐서로 가짜 이미지 생성
                z = torch.randn(self.batch_size, 64).to(self.device)
                fake_images = generator(z)  # G의 입력으로 랜덤 텐서 제공, G가 fake image 생성

                # 판별자가 가짜 이미지를 가짜로 인식하는 오차를 계산
                outputs = discriminator(fake_images)  # 가짜 이미지를 discriminator의 입력으로 제공
                d_loss_fake = criterion(outputs, fake_labels)
                fake_score = outputs

                # 진짜와 가짜 이미지를 갖고 낸 오차를 더해서 Discriminator의 오차 계산
                d_loss = d_loss_real + d_loss_fake

                # ------ Discriminator 학습 ------#
                # 역전파 알고리즘으로 Discriminator의 학습을 진행
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()  # Discriminator 학습

                # 생성자가 판별자를 속였는지에 대한 오차(Generator의 loss)를 계산
                fake_images = generator(z)
                outputs = discriminator(fake_images)  # 한번 학습한 D가 fake image를
                g_loss = criterion(outputs, real_labels)

                # ------ Generator 학습 ------#

                # 역전파 알고리즘으로 생성자 모델의 학습을 진행
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            # 학습 진행 알아보기
            print(f'Epoch [{epoch}/{self.epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, D(x): {real_score.mean().item()}, D(G(z)): {fake_score.mean().item()}')

        z = torch.randn(self.batch_size, 64).to(self.device)
        fake_images = generator(z)
        for i in range(10):
            fake_images_img = np.reshape(fake_images.data.cpu().numpy()[i], (28, 28))
            plt.imshow(fake_images_img, cmap='gray')
            plt.show()


if __name__ == '__main__':
    j = JuuGAN()
    j.process()