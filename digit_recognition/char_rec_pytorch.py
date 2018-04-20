import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms




transform = transforms.Compose([transforms.ToTensor, 
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])


trainset = torchvision.datasets.CIFAR10()

