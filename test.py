#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 13:32:54 2019

@author: amine
"""

from torchtext.vocab import Vectors
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.datasets import TranslationDataset, Multi30k
from data_utils import clean_str

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random, os, math

from sklearn.externals import joblib

from model4 import Encoder, Classifier, Model




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





TEXT = Field(tokenize= clean_str, init_token='<sos>', eos_token='<eos>')
#TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)


#train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

TEXT = Field(sequential=True, tokenize= clean_str) 
LABEL = Field(sequential=False, use_vocab=True)

datafields = [("source", TEXT),
              ("target", TEXT),
              ("plagiarism", LABEL)
              ]


train, valid = TabularDataset.splits(
               path="data", # the root directory where the data lies
               train='test_t.tsv', validation="test_v.tsv",
               format='tsv',
               skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
               fields=datafields)

#len(train[0].suspicious_text)
#len(train[0].source_text)
#train[0].plagiarism

#word2vec = joblib.load("word2vecARGensim.sav")
vectors = Vectors("wiki.ar.vec")
TEXT.build_vocab(train, vectors = vectors)
LABEL.build_vocab(train)




#print(f"Unique tokens in TEXT vocabulary: {len(LABEL.vocab)}")

train_iter, valid_iter = BucketIterator.splits(
                         (train, valid), # we pass in the datasets we want the iterator to draw data from
                         batch_size = 64,
                         device= device, # if you want to use the GPU, specify the GPU number here
                         sort_key=lambda x: len(x.source), # the BucketIterator needs to be told what function it should use to group the data.
                        )





for batch in train_iter:
    print(batch.source.size())






INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = 1
ENC_EMB_DIM = 256
ENC_HID_DIM = 512
CLS_HID_DIM = 256
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
LABEL_DIM = 1

model_l1 = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM)
model_l2 = Classifier(ENC_HID_DIM * 2, CLS_HID_DIM, OUTPUT_DIM, 0.05)
model = Model(model_l1, model_l2)




optimizer = optim.Adam(model_l2.parameters())


pad_idx = TEXT.vocab.stoi['<pad>']

criterion = nn.BCEWithLogitsLoss()



def train(model, iterator, optimizer, criterion, clip):
    
    model.train()

    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.source
        trg = batch.target
        
   
                
        optimizer.zero_grad()
        
      
        output = model(src, trg)
        
        label = batch.plagiarism.view(batch.plagiarism.size()[0], 1).float()
        

        
        loss = criterion(output, label)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)



def evaluate(model, iterator, criterion):
    
    model.eval()

    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.source
            trg = batch.target

        
            output = model(src, trg)
            
            label = batch.plagiarism.view(batch.plagiarism.size()[0], 1).float()

            loss = criterion(output, label)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)






N_EPOCHS = 10
CLIP = 10
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'tut3_model.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

for epoch in range(N_EPOCHS):
    
    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model_l1.state_dict(), MODEL_SAVE_PATH)
    
    print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} |')