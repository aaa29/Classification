#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 14:40:43 2019

@author: amine
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class Encoder(nn.Module):
    
    def __init__(self, input_dim, embed_dim, hidden_dim, dropout = 0.5, weights= None):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        if weights != None:
            self.embedding = nn.Embedding.from_pretrained(weights)
        else:    
            self.embedding = nn.Embedding(input_dim, embed_dim)
            
        self.rnn = nn.GRU(embed_dim, hidden_dim, 8, bidirectional = True) #8 is the number of layers of the rnn
        
        self.out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        #embedded [lentgh of the sebtebce, batch_size, embed_dim]
        outputs, hidden = self.rnn(embedded)
        #in the next step we'll carry only forward and backward hidden from the last layer               
        hidden = torch.cat([hidden[-1, :, :], hidden[-2, :, :]], dim = 1)
        #then apply a linear function
#        out = self.out(hidden)
        
        return hidden
    
    

class Classifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout= 0.5):
        
        super().__init__()
        
        self.in_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        
        self.classifier_l1 = nn.Linear(2, hidden_dim)
        self.classifier_l2 = nn.Linear(hidden_dim * input_dim, output_dim)
        
        
    
    def forward(self, input1, input2):
        

        input = torch.cat([input1.unsqueeze(0), input2.unsqueeze(0)])
        
        input = input.transpose(0,2)
        
        out = self.classifier_l1(input)
        out = out.view(out.size()[1], out.size()[0] * out.size()[2])
        out = self.classifier_l2(out)
        
        return out
#        return input
        
    

class Model(nn.Module):
    
    def __init__(self, encoder, classifier):
        super().__init__()
        
        self.encoder = encoder
        self.classifier = classifier

        
    def forward(self, src, trg):
        
        h1 = self.encoder(src)
        h2 = self.encoder(trg)
            

        return self.classifier(h1, h2)


