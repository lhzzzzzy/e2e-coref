import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import networkx as nx
from subprocess import Popen
import os
from utils import *
import numpy as np
from typing import Set, List
import torch.nn as nn
import random
import math


class Trainer:
    """ Class dedicated to training and evaluating the model
    """
    def __init__(self, model, train_corpus="dataset/train", val_corpus="dataset/validation"
                 , test_corpus="dataset/new_test", lr=1e-3, steps=200):

        self.__dict__.update(locals())

        self.train_corpus = os.listdir(train_corpus)
        self.val_corpus = os.listdir(val_corpus)
        
        self.model = to_cuda(model)

        self.optimizer = optim.Adam(params=[p for p in self.model.parameters()
                                            if p.requires_grad], lr=lr)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=1,
                                                   gamma=0.999)
        
        self.criterion = nn.BCELoss()
        self.steps = steps

    def train(self, num_epochs, eval_interval=5):
        """ Train a model """
        for epoch in range(1, num_epochs+1):
            self.train_epoch(epoch)

            if epoch % eval_interval == 0:
                print('\n\nEVALUATION\n\n')
                self.model.eval()
                results = self.evaluate(self.val_corpus)
                print(results)

    def train_epoch(self, epoch):
        """ Run a training epoch over 'steps' documents """
        self.model.train()
        
        batch = random.sample(self.train_corpus,self.steps)
        # batch = self.train_corpus        
        epoch_loss = []

        for document in tqdm(batch):

            with open("dataset/train/" + document, "r",encoding="gbk") as f:
                doc = json.load(f)
            # Compute loss, number gold links found, total gold links
            if doc == None:
                continue
            if doc["pronoun"]["id"] != doc["0"]["id"]:
                continue
            loss = self.train_doc(doc)
            epoch_loss.append(loss)
        

        # Step the learning rate decrease scheduler
        self.scheduler.step()
        print('Epoch: %d | Loss: %f ' \
                % (epoch, np.mean(epoch_loss)))

    def train_doc(self, document):
        """ Compute loss for a forward pass over a document """
        gold_labels = extract_gold_corefs(document)
        self.optimizer.zero_grad()
        pronoun = document["pronoun"]
        probs = self.model(document[pronoun["id"]],pronoun["indexFront"],pronoun["indexBehind"])
        loss = self.criterion(probs.squeeze(dim=1),torch.tensor(gold_labels).float())
        loss.backward()
        detached_loss = loss.detach()
        self.optimizer.step()
        
        x = detached_loss.item()
        
        if math.isnan(x):
            print(document["taskID"])
        
        return x


    def evaluate(self, eval_corpus):
        val_batch = random.sample(eval_corpus, 50)
        val_batch = eval_corpus
        tot_f, tot_r,tot_p=0,0,0
        cnt = 0
        for name in val_batch:
            with open("dataset/validation/"+name,"r",encoding="gbk") as f:
                doc = json.load(f)
                
            if doc == None:
                continue
            if doc["pronoun"]["id"] != doc["0"]["id"]:
                continue
            
            p,r,f = self.eval_doc(doc)
            tot_f += f
            tot_p += p
            tot_r += r
            cnt += 1
 
            
        return {
            'Average Precision': tot_p/cnt,
            'Average Recall': tot_r/cnt,
            'Average F1': tot_f/cnt
        }

    def eval_doc(self,doc):
        with torch.no_grad():
            pronoun = doc["pronoun"]
            probs = self.model(doc[pronoun["id"]],pronoun["indexFront"],pronoun["indexBehind"])
        
        gold_labels = extract_gold_corefs(doc)
        
        tp, pred, tot = 0,0,0
        for i in range(len(gold_labels)):
            if gold_labels[i] == 1:
                tot += 1
            if probs[i] >= 0.5:
                pred += 1
            if gold_labels[i] == 1 and probs[i] >= 0.5:
                tp += 1
        p = safe_div(tp,pred)
        r = safe_div(tp,tot)
        return p, r, safe_div(2*p*r,p+r) 

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath + '.pth')

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
        self.model = to_cuda(self.model)