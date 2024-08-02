import torchvision
import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os

#数据准备
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5, 0.5)])
train_dataset=torchvision.datasets.MNIST("./dataset",train=True,transform=transform,download=True)
train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True,drop_last=True)

#生成器
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(100,128),
            nn.Linear(128,256),
            nn.Linear(256,512),
            nn.Linear(512,28*28),
            nn.Tanh()
        )

    def forward(self,x):
        return self.model(x)
#判别器
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x=x.view(-1,28,28)
        return self.model(x)

#初始化模型，优化器及损失计算函数
#用BCEloss计算交叉熵损失
writer=SummaryWriter("./p1log")
device="cuda" if torch.cuda.is_available() else "cpu"
gen=Generator().to(device)
dis=Discriminator().to(device)
#优化器
g_optim=torch.optim.Adam(gen.parameters(),lr=1e-3)
d_optim=torch.optim.Adam(dis.parameters(),lr=1e-3)
#损失函数
loss_fn=nn.BCELoss()
epochs=20

#开始循环
for epoch in range(epochs):
    for step,(imgs,_) in enumerate(train_dataloader):
        imgs = imgs.to(device)
        size = imgs.size(0)#返回第0纬度的大小，也就是batch_size=64
        random_noise = torch.randn(size, 100, device=device)#生成64个，大小为100特征值的噪音

        # 鉴别器的优化
        d_optim.zero_grad()
        # 对真实的图片，希望判断为1
        real_output = dis(imgs)
        d_real_loss = loss_fn(real_output,
                              torch.ones_like(real_output))  # 达到鉴别器在真实图片上的损失

        # 对gen生成图片，希望对gen生成的全部判断为0
        gen_img = gen(random_noise)
        fake_output = dis(gen_img).detach()  # 截断梯度，希望判断为0
        d_fake_loss = loss_fn(fake_output,
                              torch.zeros_like(fake_output))  # 得到鉴别器在生成图片上的损失
        # 鉴别器的损失等于鉴别真实图片和鉴别生成图片的损失和
        d_loss = d_fake_loss + d_real_loss
        d_loss.backward()
        d_optim.step()

        # 生成器的优化
        g_optim.zero_grad()
        gen_img=gen(random_noise)
        fake_output = dis(gen_img)
        g_loss = loss_fn(fake_output,
                         torch.ones_like(fake_output))
        g_loss.backward()
        g_optim.step()
        #将损失记录到tensorboard
        writer.add_scalar('D_Loss_epoch:{}'.format(epoch+1), d_loss.item(), epoch * len(train_dataloader) + step)
        writer.add_scalar('G_Loss_epoch:{}'.format(epoch+1), g_loss.item(), epoch * len(train_dataloader) + step)
        #每一百次进行一次打印
        if step % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{step}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
            print("Epoch:{},生成器的损失：{}，鉴别器的损失：{}".format(epoch+1,g_loss,d_loss))
 # 每个 epoch 保存生成的图片
    with torch.no_grad():
        gen.eval()
        test_noise = torch.randn(64, 100, device=device)
        i=0
        generated_images = gen(test_noise).view(-1, 1, 28, 28)
        generated_images = (generated_images + 1) / 2  # 将 [-1, 1] 转换到 [0, 1]
        #绘图
        grid = torchvision.utils.make_grid(generated_images, nrow=8)#torchvision.utils.make_grid 将多个生成的图像（generated_images）组合成一个网格图像，nrow=8 表示每一行显示 8 张图像。
        plt.figure(figsize=(8, 8))#创建一个新的图形对象，并设置图形的大小为 8x8 英寸。
        plt.imshow(grid.cpu().numpy().transpose((1, 2, 0))) #作用是将图像的数据格式从 PyTorch 的默认格式 (C, H, W) 转换为 Matplotlib 所需的格式 (H, W, C)，以便正确显示图像。
        plt.axis('off')#关闭坐标轴和刻度
        if not os.path.exists(f'./images/epoch_{epoch+1}'):
            os.makedirs(f'./images/epoch_{epoch+1}')
        plt.savefig(f"./images/epoch_{epoch+1}/{i}.png")  # 保存生成的图片
        i+=1
        plt.close()
