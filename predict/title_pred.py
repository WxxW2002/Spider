import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_csv('data/house_with_embeddings.csv')

# df = pd.read_csv('./data/house_with_embeddings.csv')

X = data.drop(['Title', 'Subtitle', 'Total', 'Average'], axis=1)
X = X.values
feat = X.shape[1]
y = data['Average'].values

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

# 转换为Tensor
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

# 定义神经网络类
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(feat, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 实例化神经网络
net = Net()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练神经网络
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(X_train)
    loss = criterion(outputs, y_train.unsqueeze(1))
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 在测试集上进行预测
with torch.no_grad():
    predicted = net(X_test)
    mse = criterion(predicted, y_test.unsqueeze(1))
    mae = torch.mean(torch.abs(predicted - y_test.unsqueeze(1)))
    r2 = 1 - mse / torch.var(y_test.unsqueeze(1))

print('均方误差 (MSE):', mse.item())
print('平均绝对误差 (MAE):', mae.item())
print('决定系数 (R^2):', r2.item())
