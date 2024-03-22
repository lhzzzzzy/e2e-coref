import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import networkx as nx
from subprocess import Popen
import os
from utils import *
import numpy as np
from typing import Set, List
import random
torch.autograd.set_detect_anomaly(True)

def flatten(l):

    return [item for sublist in l for item in sublist]


class BaseCorefMetric:
    def __init__(self, **args):
        pass

    def calculate_p(self, keys: List, responses: List) -> float:

        raise NotImplementedError

    def calculate_r(self, keys: List, responses: List) -> float:

        raise NotImplementedError

    def calculate_f(self, keys: List, responses: List) -> float:

        p = self.calculate_p(keys, responses)
        r = self.calculate_r(keys, responses)
        if (p+r) == 0:
            return 0
        else:
            return (2 * p * r) / (p + r)

class MUC(BaseCorefMetric):
    """
    ����MUC����ָ��
    ---------------
    ver: 2022-10-25
    by: changhongyu
    """

    def calculate_p(self, keys: List[Set], responses: List[Set]) -> float:
        partitions = []
        for response in responses:
            partition = 0
            un = set()
            for key in keys:
                if response.intersection(key):
                    partition += 1
                    un = un.union(key)
            partition += len(response - un)
            partitions.append(partition)
        numerator = sum([len(response) - partition for response, partition in zip(responses, partitions)])
        denominator = sum([len(response) - 1 for response in responses])

        if denominator == 0:
            return 0.0  # or return None, depending on what you want
        else:
            return numerator / denominator


    def calculate_r(self, keys: List[Set], responses: List[Set]) -> float:
            return self.calculate_p(responses, keys)

# muc = MUC()
# muc.calculate_f(keys, responses)   # 0.4

class Trainer:
    """ Class dedicated to training and evaluating the model
    """
    def __init__(self, model, train_corpus="dataset/new_train", val_corpus="dataset/new_validation"
                 , test_corpus="dataset/new_test", lr=1e-3, steps=70):

        self.__dict__.update(locals())

        self.train_corpus = os.listdir(train_corpus)
        self.val_corpus = os.listdir(val_corpus)
        
        self.model = to_cuda(model)

        self.optimizer = optim.Adam(params=[p for p in self.model.parameters()
                                            if p.requires_grad], lr=lr)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=1,
                                                   gamma=0.999)

    def train(self, num_epochs, eval_interval=5):
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
        batch = random.sample(self.train_corpus, self.steps)

        epoch_loss = []

        for document in tqdm(batch):

            with open("dataset/new_train/" + document, "r") as f:
                doc = json.load(f)
            # Compute loss, number gold links found, total gold links
            loss = self.train_doc(doc)
            epoch_loss.append(loss)

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
                    if ((span["start"],span["end"]),(spans[i]["start"],spans[i]["end"])) in gold_corefs
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

        # loss = torch.sum(torch.log(torch.sum(torch.mul(probs, gold_indexes), dim=1).clamp_(eps, 1-eps)) * -1)
        loss = torch.sum(torch.log(torch.sum(torch.mul(probs, gold_indexes), dim=1).clamp_(eps, 1-eps)), dim=0) * -1
        

        # Backpropagate
        loss.backward()

        # Step the optimizer
        self.optimizer.step()

        return loss.item()


    def evaluate(self, eval_corpus):
        total_p = 0.0
        total_r = 0.0
        total_f = 0.0

        muc_metric = MUC()

        for document in tqdm(eval_corpus):
            with open("dataset/new_validation/" + document, "r", encoding="utf-8") as f:
                doc = json.load(f)

            keys, responses = self.evaluate_doc(doc)

            p = muc_metric.calculate_p(keys, responses)
            r = muc_metric.calculate_r(keys, responses)
            f = muc_metric.calculate_f(keys, responses)

            total_p += p
            total_r += r
            total_f += f
            
            if "45" in document:
                break

        avg_p = total_p / len(eval_corpus)
        avg_r = total_r / len(eval_corpus)
        avg_f = total_f / len(eval_corpus)

        return {
            'Average Precision': avg_p,
            'Average Recall': avg_r,
            'Average F1': avg_f
        }

    def evaluate_doc(self, document):
        gold_corefs, gold_mentions = extract_gold_corefs(document)

        responses = self.predict(document)

        keys = []
        span_idx = {}
        cnt = 0
        for coref in gold_corefs:
            if coref[0] not in span_idx:
                span_idx[coref[0]] = cnt
                keys.append(set())
                cnt += 1
                
            
            idx = span_idx[coref[0]]
            keys[idx].add(coref[1])
                
        return keys, responses


    def predict(self, doc):
        """ Predict coreference clusters in a document """

        # Set to eval mode
        self.model.eval()

        # Pass the document through the model
        spans, probs = self.model(doc["sentence"])

        span_clusters = []
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