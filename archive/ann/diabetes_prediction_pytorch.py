import numpy as np
import torch
from torch.autograd import Variable
import torch.functional as F


file_loc = r'D:\dev\data\diabetes.csv.gz'
data = np.loadtxt(file_loc, delimiter=',', dtype=np.float32)


x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0]]))
y_data = Variable(torch.Tensor([[2.0],[4.0],[6.0]]))
w = Variable(torch.Tensor([1.0]), requires_grad=True)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat

    
model = Model()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

for epoch in range(2000):
    y_hat = model(x_data)
    loss = criterion(y_hat, y_data)
    print(epoch, loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


output = Variable(torch.Tensor([[4.0]]))
y_pred = model(output)
print("Prediction = ", y_pred.data[0])






