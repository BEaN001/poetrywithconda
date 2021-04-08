import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sklearn import preprocessing


class Net(nn.Module):
    def __init__(self, in_count, out_count):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_count, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, out_count)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return self.softmax(x)


df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/iris.csv",
    na_values=['NA', '?'])

le = preprocessing.LabelEncoder()

x = df[['sepal_l', 'sepal_w', 'petal_l', 'petal_w']].values
y = le.fit_transform(df['species'])
classes = le.classes_

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

x_train = Variable(torch.Tensor(x_train).float())
x_test = Variable(torch.Tensor(x_test).float())
y_train = Variable(torch.Tensor(y_train).long())
y_test = Variable(torch.Tensor(y_test).long())

net = Net(x.shape[1], len(classes))

criterion = nn.CrossEntropyLoss()  # cross entropy loss

# optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    out = net(x_train)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, loss: {loss.item()}")

pred_prob = net(x_test)
_, pred = torch.max(pred_prob, 1)

correct = accuracy_score(y_test,pred)
print(f"Accuracy: {correct}")