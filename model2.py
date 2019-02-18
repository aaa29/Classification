#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:20:55 2019

@author: amine
"""



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class Encoder(nn.Module):
    
    def __init__(self, input_dim, embed_dim, hidden_dim, output_dim, dropout = 0.5, weights = None):
        
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = input_dim

        
        try:
            self.embedding = nn.Embedding.from_pretrained(weights)
        except:    
            self.embedding = nn.Embedding(input_dim, embed_dim)
            
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        
    
    def forward(self, x):
        
        #x : [batch_size, input_size(vocab_size)]
        x = x.permute(1, 0)
#        print("x", x.size())

        #unsqueeze(1) in order to add the number of channels with parameters
        embedded = self.embedding(x)
#        print("embedded", embedded.size())
        out, _ = self.lstm(embedded)
        out = out[:, -1 ,:].squeeze(1)

        
        return out
        
        
        

class CNN(nn.Module):
    
    def __init__(self, input_dim, embed_dim, nb_kernels, kernel_size, output_dim, dropout = 0.5, weights = None):
        
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = input_dim
        self.nb_kernels = nb_kernels
        self.kernel_size = kernel_size
        
        try:
            self.embedding = nn.Embedding.from_pretrained(weights)
        except:    
            self.embedding = nn.Embedding(input_dim, embed_dim)
            

#        self.conv = [nn.Conv2d(1, nb_kernels, (kernel_size, embed_dim)) for i in range(nb_kernels)]
        self.conv = nn.Conv2d(1, nb_kernels, (kernel_size, embed_dim))
        
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(nb_kernels, output_dim)
        
    
    def forward(self, x):
        #x : [batch_size, input_size(vocab_size)]
        x = x.permute(1, 0)
        

        #unsqueeze(1) in order to add the number of channels with parameters
        embedded = self.embedding(x).unsqueeze(1)
        

        
        
#        convolved = [F.relu(self.conv[i](embedded).squeeze(3)) for i in range(self.nb_kernels)]
        convolved = F.relu(self.conv(embedded).squeeze(3))
#        print(convolved.size())

        
#        pooled = [F.max_pool1d(convolved[i], convolved[i].shape[2]).squeeze(2) for i in range(self.nb_kernels)]
        pooled = F.max_pool1d(convolved, convolved.shape[2]).squeeze(2)
#        print(pooled.size())
        


        
        return self.dropout(pooled)
    
    
    
class Classifier(nn.Module):
    
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout= 0.5):
        
        super().__init__()
        

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        
        
        
        self.classifier_l1 = nn.Linear(2, hidden_dim)
        self.classifier_l2 = nn.Linear(hidden_dim * input_dim, output_dim)
        self.softmax = nn.Softmax(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, input1, input2):
        
#        print(input2.size(), input1.size())
        

        

        
        in1 = torch.cat([input1.unsqueeze(1), input2.unsqueeze(1)], 1)
        
        in1 = in1.transpose(1,2)

        
        
#        input = input.transpose(0,2)
        
        out = self.dropout(self.classifier_l1(in1))
        out = out.view([out.size()[0], out.size()[1] * out.size()[2]])
#        print("out", out)
        
        

        out = self.classifier_l2(out)
        
        return self.dropout(out)
#        return input
        
    

class Classifier_2(nn.Module):
    
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout= 0.5):
        
        super().__init__()
        

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        
        
        
        self.classifier_l1 = nn.Linear(input_dim*2, hidden_dim)
        self.classifier_l2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        
    
    def forward(self, input1, input2):
        
#        print(input2.size(), input1.size())
        

        in1 = torch.cat([input1, input2], 1)
#        print("in1", in1.size())
        

        out = self.dropout(self.classifier_l1(in1))

        out = self.classifier_l2(out)
        
        return self.dropout(out)
#        return input
        
    
#in1 = torch.rand(64, 100)
#in2 = torch.rand(64, 100)
#
#in3 = F.cosine_similarity(in1, in2)