from gcommand_dataset_cnn import GCommandLoader
import torch
from torch import optim
import torch.nn as nn
from torch.functional import F
import sys

def train(model, optimizer, train_loader, val_loader, epochs=10):
    global device
    train_loss = 0
    train_correct = 0
    for e in range(epochs):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            loss = F.nll_loss(input=output, target=labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.max(dim=1, keepdim=True)[1]  # get the index of the max log-probability
            train_correct += pred.eq(labels.view_as(pred)).cpu().sum().item()

        train_loss /= len(train_loader.dataset)
        train_correct /= len(train_loader.dataset)
        if val_loader:
            val_loss, val_acc = test(model=model, loader=val_loader)
        else:
            val_loss, val_acc = None, None

        print(f'Epoch: {e + 1} [{(e + 1)}/{epochs}] Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')
        print(f'Epoch: {e + 1} [{(e + 1)}/{epochs}] Train ACC:  {train_correct:.3f},  Val ACC:  {val_acc:.3f}')

def test(model, loader):
    global device
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(dim=1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).cpu().sum().item()

    loss /= len(loader.dataset)
    return loss, correct / len(loader.dataset)

def predict(model, loader):
    preds = []
    model.eval()
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)

            output = model(data)
            preds.append(output.max(dim=1, keepdim=True)[1])

    return torch.cat(preds, dim=0).detach()

def _make_layers(cfg):
    layers = []
    in_channels = 1
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        arch = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

        self.conv = _make_layers(arch)
        self.fc1 = nn.Linear(7680, 512)
        self.fc2 = nn.Linear(512, 30)

    def forward(self, x):
        x = self.conv(x)
     
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

train_set = GCommandLoader('gcommands/train')
val_set = GCommandLoader('gcommands/valid')
test_set = GCommandLoader('gcommands/test')

batch_size = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

print(f'device: {device}')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

print(len(train_loader.dataset))
print(len(val_loader.dataset))
print(len(test_loader.dataset))

epochs = 7

model = CNN()
model.to(device)

adam = optim.Adam(model.parameters(), lr=0.0001)
train(model=model, optimizer=adam, train_loader=train_loader, val_loader=val_loader, epochs=epochs)

y_hat = predict(model=model, loader=test_loader)

index_2_classes = {i:c for c, i in train_loader.dataset.class_to_idx.items()}
if sys.platform == 'linux':
    X = [x.rsplit('/', 1)[1] for x, y in test_loader.dataset.spects]
else:
    X = [x.rsplit('\\', 1)[1] for x, y in test_loader.dataset.spects]
output = [f'{x},{index_2_classes[y.item()]}\n' for x, y in zip(X, y_hat)]
output = sorted(output, key=lambda x: int(x.split('.')[0]))
with open('test_y', 'w') as f:
    f.writelines(output)
