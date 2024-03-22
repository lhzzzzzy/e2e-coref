import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import networkx as nx
from subprocess import Popen
import os
from utils import *
import numpy as np

class Trainer:
    """ Class dedicated to training and evaluating the model"""
    
    def __init__(self, model, train_corpus="dataset/new_train/", val_corpus="dataset/new_validation/"
                 , test_corpus="dataset/new_test/", lr=1e-3, steps=100):

        self.__dict__.update(locals())

        self.train_corpus = os.listdir(train_corpus)
        self.val_corpus = os.listdir(val_corpus)
        
        self.model = to_cuda(model)

        self.optimizer = optim.Adam(params=[p for p in self.model.parameters()
                                            if p.requires_grad], lr=lr)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=100,
                                                   gamma=0.001)

    def train(self, num_epochs, eval_interval=100):
        """ Train a model """
        for epoch in range(1, num_epochs+1):
            self.train_epoch(epoch)

            # Save often
            # self.save_model(str(datetime.now())) 

            # Evaluate every eval_interval epochs
            if epoch % eval_interval == 0:
                print('\n\nEVALUATION\n\n')
                self.model.eval()
                results = self.evaluate(self.val_corpus)
                print(results)

    def train_epoch(self, epoch):
        """ Run a training epoch over 'steps' documents """

        # Set model to train (enables dropout)
        self.model.train()

        # Randomly sample documents from the train corpus
        # batch = random.sample(self.train_corpus, self.steps)
        batch = self.train_corpus

        epoch_loss = []

        for document in tqdm(batch):

            with open("dataset/new_train/" + document, "r") as f:
                doc = json.load(f)
            # Compute loss, number gold links found, total gold links
            loss = self.train_doc(doc)
            epoch_loss.append(loss)
            break


        # Step the learning rate decrease scheduler
        self.scheduler.step()

        print('Epoch: %d | Loss: %f ' \
                % (epoch, np.mean(epoch_loss)))

    def train_doc(self, document):
        """ Compute loss for a forward pass over a document """

        # Extract gold coreference links
        gold_corefs, gold_mentions= extract_gold_corefs(document)

        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        # Init metrics
        # mentions_found, corefs_found, corefs_chosen = 0, 0, 0

        # Predict coref probabilites for each span in a document
        spans, probs = self.model(document["sentence"])

        # Get log-likelihood of correct antecedents implied by gold clustering
        gold_indexes = to_cuda(torch.zeros_like(probs))
        for idx, span in enumerate(spans):
            
            # Log number of mentions found
            if (span["start"], span["end"]) in gold_mentions:
                # mentions_found += 1

                # Check which of these tuples are in the gold set, if any
                golds = [
                    i for i in span["pre_spans"]
                    if ((span["start"],span["end"]),(span[i]["start"],span[i]["end"])) in gold_corefs
                ]

                # If gold_pred_idx is not empty, consider the probabilities of the found antecedents
                if golds:
                    gold_indexes[idx, golds] = 1
                    # Progress logging for recall
                else:
                    # Otherwise, set gold to dummy
                    gold_indexes[idx, len(span["pre_spans"])] = 1

        # Negative marginal log-likelihood
        eps = 1e-8

        loss = torch.sum(torch.log(torch.sum(torch.mul(probs, gold_indexes), dim=1).clamp_(eps, 1-eps)) * -1)

        # Backpropagate
        loss.backward()
        
        print(self.predict(document))

        # Step the optimizer
        self.optimizer.step()

        return loss.item()

    def evaluate(self, val_corpus, eval_script='../src/eval/scorer.pl'):
        """ Evaluate a corpus of CoNLL-2012 gold files """

        # Predict files
        print('Evaluating on validation corpus...')
        
        predicted_docs = []
        for f_name in tqdm(val_corpus):
            with open("dataset/new_validation/" + f_name, "r") as f:
                data = json.load(f)
            predicted_docs.append(self.predict(data))

        # TODO: calculate f1
        
        return results

    def predict(self, doc):
        """ Predict coreference clusters in a document """

        # Set to eval mode
        self.model.eval()

        # Initialize graph (mentions are nodes and edges indicate coref linkage)
        graph = nx.Graph()

        # Pass the document through the model
        spans, probs = self.model(doc["sentence"])

        span_clusters = {}
        # Cluster found coreference links
        for i, span in enumerate(spans):

            # Loss implicitly pushes coref links above 0, rest below 0
            found_corefs = [idx
                            for idx, _ in enumerate(span["pre_spans"])
                            if probs[i, idx] > probs[i, len(span["pre_spans"])]]

            # If we have any
            if any(found_corefs):

                # Add edges between all spans in the cluster
                cluster = set()
                for coref_idx in found_corefs:
                    link = spans[coref_idx]
                    cluster.add((link["start"], link["end"]))
                span_clusters.append(cluster)

        return span_clusters
    

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath + '.pth')

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
        self.model = to_cuda(self.model)