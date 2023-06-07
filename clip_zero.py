import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10

import clip


def count_parameter(model):
    return sum(p.numel() for p in model.parameters())


def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).cuda()
            
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            
            zeroshot_weights.append(class_embedding)
            
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='RN50', help='clip backbone')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--file_name', type=str, default='imagenet.txt', help='prompt text file')
    
    args = parser.parse_args()
    
    model_name = args.model
    batch_size = args.batch_size
    file_name = args.file_name
    
    print(model_name, batch_size, file_name)
    
    prompt = []
    
    with open(file_name, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            prompt.append(line.strip())
    
    available_model = clip.available_models()
    
    assert model_name in available_model
    
    model, preprocess = clip.load(model_name)
    model.cuda().eval()
    
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

#     print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
#     print("Input resolution:", input_resolution)
#     print("Context length:", context_length)
#     print("Vocab size:", vocab_size)
#     print(preprocess)
    
    cifar100 = CIFAR100('./cifar100_data', train=False, transform=preprocess, download=True)
    cifar10 = CIFAR10('./cifar10_data', train=False, transform=preprocess, download=True)
    
    cifar10_dataloader = DataLoader(cifar10, batch_size=batch_size, shuffle=False, num_workers=2)
    cifar100_dataloader = DataLoader(cifar100, batch_size=batch_size, shuffle=False, num_workers=2)
    
    
    with torch.no_grad():
        
        zeroshot_weights = zeroshot_classifier(cifar10.classes, prompt)
        top1, top5, n = 0., 0., 0.
        
        for i, (images, target) in enumerate(tqdm(cifar10_dataloader)):
            images, target = images.cuda(), target.cuda()
            
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights
            
            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            
        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100
        
        with open('./results.txt', 'a') as f:
            f.write('\n')
            f.write('---------- cifar10 ----------\n')
            if file_name != 'imagenet.txt':
                for p in prompt:
                    f.write(f'{p.format("[CLASS]")} \n')
            else:
                f.write('imagenet prompt\n')
                
            f.write(f'model: {model_name}  param: {count_parameter(model.visual)}  top1: {top1:.2f}  top5: {top5:.2f} \n')
        
        zeroshot_weights = zeroshot_classifier(cifar100.classes, prompt)
        top1, top5, n = 0., 0., 0.
        
        for i, (images, target) in enumerate(tqdm(cifar100_dataloader)):
            images, target = images.cuda(), target.cuda()
            
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights
            
            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            
        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100
        
        with open('./results.txt', 'a') as f:
            f.write('\n')
            f.write('---------- cifar100 ----------\n')
            if file_name != 'imagenet.txt':
                for p in prompt:
                    f.write(f'{p.format("[CLASS]")} \n')
            else:
                f.write('imagenet prompt\n')
                
            f.write(f'model: {model_name}  param: {count_parameter(model.visual)}  top1: {top1:.2f}  top5: {top5:.2f} \n')