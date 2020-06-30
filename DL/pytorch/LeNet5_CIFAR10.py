import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # x:[b,3,32,32] => [b,6,28,28]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        # x: [b,3,32,32] => [b,16,5,5]
        batchsz = x.size(0)
        x = self.conv_unit(x)
        # [b,16,5,5] => [b,16*5*5]
        x = x.view(batchsz, 16*5*5)
        # [b,16*5*5] => [b,10]
        out = self.fc_unit(x)
        return out


def main():
    batchsz = 32

    # step1: load data
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    # step2 model and criteon
    device = torch.device('cuda')
    model = LeNet5().to(device)
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
