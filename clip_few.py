import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10

import clip

def count_parameter(model):
    return sum(p.numel() for p in model.parameters())


def get_correct_samples(scores: torch.Tensor, labels: torch.Tensor) -> int:
    """Gets the number of correctly classified examples.

    Args:
        scores: the scores predicted with the network.
        labels: the class labels.

    Returns: 
        the number of correct samples.
    """
    classes_predicted = torch.argmax(scores, 1)
    return (classes_predicted == labels).sum().item()


def set_requires_grad_for_layer(layer: torch.nn.Module, train: bool) -> None:
    """Sets the attribute requires_grad to True or False for each parameter.
        
        Args:
            layer: the layer to freeze.
            train: if true train the layer.
    """
    for p in layer.parameters():
        p.requires_grad = train

        
def training_loop(model, optimizer, criterion, t_loader, v_loader):
    model.train()
    
    
    for i, data in enumerate(t_loader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
    
    
    model.eval()
    
    _loss= 0
    _correct = 0
    _total= 0
    
    linear_probe_model.eval()
    
    for i, data in enumerate(v_loader):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = linear_probe_model(inputs)

            loss = criterion(outputs, labels)

            _loss += loss.item()
            _total += labels.size(0)
            _correct += get_correct_samples(outputs, labels)
            
    return _loss / _total, _correct / _total * 100


def execute(model, optimizer, criterion, epochs, t_loader, v_loader, model_name, dataset_name):
    best_epoch = 0
    best_acc = 0
    
    for epoch in tqdm(range(epochs)):
        loss, acc = training_loop(model, optimizer, criterion, t_loader, v_loader)
        
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch
            
            
    with open('./results2.txt', 'a') as f:
        f.write(f'{model_name}   data: {dataset_name}   acc: {best_acc}   epoch: {best_epoch}')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='RN50', help='clip backbone')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='batch size')
    
    args = parser.parse_args()
    
    model_name = args.model
    batch_size = args.batch_size
    epochs = args.epoch
    
    print(model_name, batch_size)
    
    available_model = clip.available_models()
    
    assert model_name in available_model
    
    end = 0
    
    if model_name == 'RN50':
        end = 1024
    elif model_name == 'RN101':
        end = 512
    elif model_name == 'RN50x4':
        end = 640
    elif model_name == 'RN50x16':
        end = 768
    elif model_name == 'RN50x64':
        end = 1024
    
    model, preprocess = clip.load(model_name)
    model.cuda().eval()
    
    cifar100_t = CIFAR100('./cifar100_data', train=True, transform=preprocess, download=True)
    cifar10_t = CIFAR10('./cifar10_data', train=True, transform=preprocess, download=True)
    
    cifar10_dataloader_t = DataLoader(cifar10_t, batch_size=batch_size, shuffle=False, num_workers=2)
    cifar100_dataloader_t = DataLoader(cifar100_t, batch_size=batch_size, shuffle=False, num_workers=2)
    
    cifar100_v = CIFAR100('./cifar100_data', train=False, transform=preprocess, download=True)
    cifar10_v = CIFAR10('./cifar10_data', train=False, transform=preprocess, download=True)
    
    cifar10_dataloader_v = DataLoader(cifar10_v, batch_size=batch_size, shuffle=False, num_workers=2)
    cifar100_dataloader_v = DataLoader(cifar100_v, batch_size=batch_size, shuffle=False, num_workers=2)
    
    linear_probe_model = nn.Sequential(
        model.visual,
        nn.Linear(end, len(cifar10_t.classes), dtype=torch.float16)
    )
    
    linear_probe_model = linear_probe_model.cuda().eval()
    
    set_requires_grad_for_layer(linear_probe_model[0], False)
    set_requires_grad_for_layer(linear_probe_model[1], True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(linear_probe_model.parameters(), lr=1e-4, momentum=0.9)
    
    execute(linear_probe_model, optimizer, criterion, epochs , cifar10_dataloader_t, cifar10_dataloader_v, model_name, 'cifar10')
                
                
    linear_probe_model = nn.Sequential(
        model.visual,
        nn.Linear(end, len(cifar100_t.classes), dtype=torch.float16)
    )
    
    linear_probe_model = linear_probe_model.cuda().eval()
    
    set_requires_grad_for_layer(linear_probe_model[0], False)
    set_requires_grad_for_layer(linear_probe_model[1], True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(linear_probe_model.parameters(), lr=1e-4, momentum=0.9)
    
    execute(linear_probe_model, optimizer, criterion, epochs, cifar100_dataloader_t, cifar100_dataloader_v, model_name, 'cifar100')