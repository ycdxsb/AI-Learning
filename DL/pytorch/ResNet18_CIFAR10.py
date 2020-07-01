import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim


class ResNetBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(
            ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if(ch_out != ch_in):
            # make out and x the same size
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        # [b,ch,wh,w]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        out = self.extra(x)+out
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # 4 block
        # [b,64,h,w] => [b,128,h,w]
        self.block1 = ResNetBlock(64, 128,stride=2)
        # [b,128,h,w] => [b,256,h,w]
        self.block2 = ResNetBlock(128, 256,stride=2)
        # [b,256,h,w] => [b,512,h,w]
        self.block3 = ResNetBlock(256, 512,stride=2)
        # [b,256,h,w] => [b,1024,h,w]
        self.block4 = ResNetBlock(512, 512,stride=2)

        self.outlayer = nn.Linear(512, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        # [b,64,h,w] => [b,1024,h,w]
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = F.adaptive_avg_pool2d(out,[1,1])
        out = out.view(out.size(0),-1)
        out = self.outlayer(out)
        return out


def main():
    batchsz = 32

    # step1: load data
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    # step2 model and criteon
    device = torch.device('cuda')
    model = ResNet18().to(device)
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1000):
        # step3 train
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)
            out = model(x)
            loss = criteon(out, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)
            print("acc: %d/%d" % (total_correct, total_num))


if __name__ == "__main__":
    main()
