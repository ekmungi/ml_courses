
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable


# ## Load data

# In[2]:


train_dataset = datasets.MNIST(root='D:/dev/data/mnist', train=True, transform=transforms.ToTensor(), download=True)
print(len(train_dataset))


# In[10]:


print(train_dataset[0][0].size(), train_dataset[0][1])


# In[3]:


test_dataset = datasets.MNIST(root='D:/dev/data/mnist', train=False, transform=transforms.ToTensor(), download=True)
print(len(test_dataset))


# ## Display MNIST

# In[11]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


print(train_dataset[0][0].numpy().shape)


# In[13]:


show_img = train_dataset[0][0].numpy().reshape(28,28)


# In[15]:


plt.imshow(show_img, cmap='gray')


# In[17]:


# Label
print(train_dataset[0][1].numpy())


# ## Make dataset iterable

# In[18]:


len(train_dataset)


# In[19]:


batch_size = 100


# In[20]:


n_iters = 3000


# In[22]:


n_epochs = int(n_iters / (len(train_dataset)/batch_size))
print(n_epochs)


# ### Create iterable object : Training dataset

# In[23]:


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# ### Check iterability

# In[24]:


import collections


# In[25]:


isinstance(train_loader, collections.Iterable)


# ### Create iterable object : Test dataset

# In[26]:


test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# ### Check iterability

# In[27]:


isinstance(test_loader, collections.Iterable)


# ## Build model

# In[29]:


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features=input_dims, out_features=output_dims)
        
    def forward(self, x):
        return self.linear(x)


# ## Instatiate model

# In[30]:


input_dims = 28*28
output_dims = 10
logistic_reg = LogisticRegressionModel(input_dims, output_dims)


# ## Instantiate loss class

# In[31]:


criterion = nn.CrossEntropyLoss()


# ## model.parameters() explained

# In[36]:


print(logistic_reg.parameters())
print(len(list(logistic_reg.parameters())))

print(list(logistic_reg.parameters())[0].size())
print(list(logistic_reg.parameters())[1].size())


# ## Optimizer

# In[42]:


learning_rate = 0.001
optimizer = torch.optim.SGD(logistic_reg.parameters(), lr=learning_rate)


# ## Training phase

# In[44]:


iteration = 0
for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1,28*28))
        labels = Variable(labels)
        
        optimizer.zero_grad()
        
        outputs = logistic_reg(images)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        iteration += 1
        
        if iteration%500 == 0:
            correct = 0
            total = 0
            
            for images, labels in test_loader:
                images = Variable(images.view(-1,28*28))
                labels = Variable(labels)
                
                outputs = logistic_reg(images)
                
                print(outputs[0].numpy())
                
                predicted = torch.argmax(outputs, dim=1)
                
                total += labels.size()[0]
                
                correct += (predicted == labels).sum()
                
                
            accuracy = 100*correct/total
            
            print('Iteration: {}, Loss: {}, Accuracy: {}%'.format(iteration, loss, accuracy))
            
    break
                

