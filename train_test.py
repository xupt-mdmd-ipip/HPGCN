import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from model import *
from function import *

device = torch.device("cuda:0")


def train_nn(train_ds, epoch):
    model_1 = model_nn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_1.parameters(), lr=0.0001)
    total_loss = 0
    y_train = train_ds.label
    train_dl = DataLoader(train_ds, batch_size=16)
    trainloss = []
    acc_all = []
    index = 1

    for epoch in range(epoch):
        train_pred = []
        for i, (data, label) in enumerate(train_dl):
            input = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs, _ = model_1(input)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            train_pred = np.concatenate((train_pred, outputs))
        acc_c = accuracy_score(y_train, train_pred, normalize=True)
        acc_all.append(acc_c)
        trainloss.append(loss.item())
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (
        epoch + 1, total_loss / (epoch + 1), loss.item()))
        print('[Train acc: %4f]' % acc_c)
        with open("loss/train_loss.txt", 'w') as train_loss:
            train_loss.write(str(trainloss))
        if (epoch + 1) % 1000 == 0:
            save_net(model_1, 'train_' + str(index) + 'k')
            index = index + 1


# 测试
def test_nn(test_ds, net_path):
    model = torch.load(net_path)
    model.eval()
    test_dl = DataLoader(test_ds)
    y_test = test_ds.label
    test_pred = []

    for data, _ in test_dl:
        input = data.to(device)
        outputs, _ = model(input)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        test_pred = np.concatenate((test_pred, outputs))

    acc_c = classification_report(y_test, test_pred, digits=4)
    print(acc_c)
    with open("./test_acc_1k.txt", 'w') as acc_test:
        acc_test.write(str(acc_c))
